import argparse
import asyncio
import json
import os
import re

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics import ContextPrecision, ContextRecall, ContextEntityRecall, Faithfulness, NoiseSensitivity, AnswerRelevancy
from tqdm import tqdm

load_dotenv()

evaluator_client = AsyncOpenAI(
    base_url=os.getenv('EVALUATOR_LLM_BASE_URL') or os.getenv('EVALUATOR_GENERATOR_LLM_BASE_URL'),
    api_key=os.getenv('EVALUATOR_LLM_API_KEY'),
    timeout=float(os.getenv("EVALUATOR_LLM_REQUEST_TIMEOUT", "120")),
)

evaluator_model = os.getenv('EVALUATOR_LLM_MODEL_NAME')
if not evaluator_model:
    raise ValueError("EVALUATOR_LLM_MODEL_NAME not found in .env file")

EVALUATOR_TEMPERATURE = float(os.getenv("EVALUATOR_LLM_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0")))
EVALUATOR_TOP_P = float(os.getenv("EVALUATOR_LLM_TOP_P", os.getenv("LLM_TOP_P", "1")))
EVALUATOR_MAX_TOKENS = int(os.getenv("EVALUATOR_LLM_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "512")))
EVALUATOR_REQUEST_TIMEOUT = float(os.getenv("EVALUATOR_LLM_REQUEST_TIMEOUT", "120"))

evaluator_llm = llm_factory(
    evaluator_model,
    client=evaluator_client,
    temperature=EVALUATOR_TEMPERATURE,
    top_p=EVALUATOR_TOP_P,
    max_tokens=EVALUATOR_MAX_TOKENS,
)

embeddings_client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    client=embeddings_client
)


class RagasEmbeddingsAdapter:
    def __init__(self, base_embeddings):
        self.base_embeddings = base_embeddings

    def embed_query(self, text):
        return self.base_embeddings.embed_text(text)

    def embed_documents(self, texts):
        return self.base_embeddings.embed_texts(texts)

metrics = [
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
    ContextEntityRecall(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    NoiseSensitivity(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=RagasEmbeddingsAdapter(embeddings)),
]

metric_lookup = {metric.name: metric for metric in metrics}

parser = argparse.ArgumentParser(description="Evaluate generated answers with ragas metrics.")
parser.add_argument("--input", default="03_with_answers.csv", help="Input CSV containing question, context, ground truth, and generated answer.")
parser.add_argument("--output", default="04_evaluated.csv", help="Output CSV path.")
parser.add_argument("--start", type=int, default=0, help="Start row index (0-based).")
parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to evaluate.")
parser.add_argument("--concurrency", type=int, default=1, help="Number of rows to evaluate in parallel.")
parser.add_argument(
    "--metrics",
    default="all",
    help="Comma-separated metric names to evaluate, or 'all'.",
)
args = parser.parse_args()

source_df = pd.read_csv(args.input)
if args.start < 0:
    raise ValueError("--start must be >= 0")

if args.limit is not None and args.limit < 1:
    raise ValueError("--limit must be >= 1")
if args.concurrency < 1:
    raise ValueError("--concurrency must be >= 1")

if args.metrics.strip().lower() == "all":
    active_metrics = metrics
else:
    selected_metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    unknown_metrics = [name for name in selected_metric_names if name not in metric_lookup]
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {', '.join(unknown_metrics)}")
    active_metrics = [metric_lookup[name] for name in selected_metric_names]

end_idx = None if args.limit is None else args.start + args.limit
df = source_df.iloc[args.start:end_idx].copy()
target_indices = list(df.index)
OUTPUT_CSV = args.output
existing_output_df = None
metric_timeout = os.getenv("EVAL_METRIC_TIMEOUT")
METRIC_TIMEOUT_SECONDS = float(metric_timeout) if metric_timeout else None

RETRY_LIMITS = [
    {"question": 400, "answer": 1600, "reference": 1600},
    {"question": 300, "answer": 900, "reference": 900},
    {"question": 220, "answer": 600, "reference": 600},
    {"question": 180, "answer": 450, "reference": 450},
    {"question": 140, "answer": 320, "reference": 320},
    {"question": 120, "answer": 220, "reference": 220},
]

print(
    f"Evaluating {len(df)} answers from row {args.start} "
    f"with concurrency={args.concurrency}, metrics={[metric.name for metric in active_metrics]}, "
    f"temperature={EVALUATOR_TEMPERATURE}, top_p={EVALUATOR_TOP_P}, "
    f"max_tokens={EVALUATOR_MAX_TOKENS}, request_timeout={EVALUATOR_REQUEST_TIMEOUT}s..."
)


def is_auth_error(error):
    error_text = str(error).lower()
    return "invalid_api_key" in error_text or "incorrect api key provided" in error_text


def is_length_error(error):
    return "max_tokens length limit" in str(error).lower()


def truncate_text(value, limit):
    if pd.isna(value):
        return ""

    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text

    return text[: limit - 3].rstrip() + "..."


def compress_text(value, limit, max_sentences=3):
    text = truncate_text(value, max(limit * 2, limit))
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    compact = " ".join(sentences[:max_sentences]).strip()
    return truncate_text(compact or text, limit)


def extract_metric_value(value):
    if hasattr(value, "value"):
        return value.value
    return value


def build_inputs(question, answer, reference, contexts, limits):
    clean_question = compress_text(question, limits["question"], max_sentences=2)
    clean_answer = compress_text(answer, limits["answer"], max_sentences=4)
    clean_reference = compress_text(reference, limits["reference"], max_sentences=4)
    context_limit = limits.get("context", 800)
    clean_contexts = [compress_text(c, context_limit, max_sentences=4) for c in contexts]
    return clean_question, clean_answer, clean_reference, clean_contexts


async def evaluate_metric(metric, question, answer, reference, contexts, limits):
    clean_question, clean_answer, clean_reference, clean_contexts = build_inputs(
        question, answer, reference, contexts, limits
    )

    sample_kwargs = {
        "user_input": clean_question,
        "response": clean_answer,
    }
    if metric.name in ("context_precision", "context_recall", "context_entity_recall", "faithfulness", "noise_sensitivity"):
        sample_kwargs["retrieved_contexts"] = clean_contexts
    if metric.name in ("context_precision", "context_recall", "context_entity_recall", "noise_sensitivity"):
        sample_kwargs["reference"] = clean_reference

    score = await metric.single_turn_ascore(SingleTurnSample(**sample_kwargs), timeout=METRIC_TIMEOUT_SECONDS)

    return extract_metric_value(score)


def get_metric_retry_limits(metric_name):
    if metric_name in ("context_precision", "context_recall", "context_entity_recall", "noise_sensitivity"):
        return [
            {"question": 260, "answer": 800, "reference": 900, "context": 1000},
            {"question": 200, "answer": 600, "reference": 700, "context": 800},
            {"question": 160, "answer": 450, "reference": 500, "context": 600},
            {"question": 120, "answer": 320, "reference": 320, "context": 450},
        ]
    if metric_name == "faithfulness":
        return RETRY_LIMITS
    if metric_name == "answer_relevancy":
        return [
            {"question": 260, "answer": 900, "reference": 0},
            {"question": 200, "answer": 600, "reference": 0},
            {"question": 160, "answer": 400, "reference": 0},
            {"question": 120, "answer": 260, "reference": 0},
        ]
    return RETRY_LIMITS


def parse_contexts(value):
    if pd.isna(value) or value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    if "\n\n---\n\n" in text:
        return [t.strip() for t in text.split("\n\n---\n\n") if t.strip()]
    return [text]


async def evaluate_row(idx, question, answer, reference, contexts, existing_result):
    results = {metric.name: existing_result.get(metric.name) for metric in active_metrics}

    if pd.isna(answer) or str(answer).strip() == '':
        return results

    for metric in active_metrics:
        if results.get(metric.name) is not None:
            continue

        last_error = None
        if metric.name in ("context_precision", "context_recall", "context_entity_recall", "noise_sensitivity", "faithfulness") and not contexts:
            continue

        for limits in get_metric_retry_limits(metric.name):
            try:
                results[metric.name] = await evaluate_metric(
                    metric, question, answer, reference, contexts, limits
                )
                break
            except Exception as e:
                if is_auth_error(e):
                    raise RuntimeError(
                        "Authentication failed for the evaluator or embeddings client. "
                        "Check OPENAI_API_KEY and evaluator credentials in .env."
                    ) from e
                last_error = e
                if not is_length_error(e):
                    break

        if results.get(metric.name) is None and last_error is not None:
            print(f"Error evaluating row {idx} for {metric.name}: {last_error}")

    return results


def load_existing_results():
    global existing_output_df
    if not os.path.exists(OUTPUT_CSV):
        existing_output_df = source_df.copy()
        return [{} for _ in range(len(df))]

    existing_df = pd.read_csv(OUTPUT_CSV)
    existing_output_df = existing_df.copy()
    existing_results = []
    for target_idx in target_indices:
        row = existing_df.iloc[target_idx]
        row_result = {}
        for metric in active_metrics:
            value = row.get(metric.name)
            if pd.isna(value):
                row_result[metric.name] = None
            else:
                try:
                    row_result[metric.name] = float(value)
                except (TypeError, ValueError):
                    row_result[metric.name] = value
        existing_results.append(row_result)
    return existing_results


def persist_results(results):
    output_df = existing_output_df.copy() if existing_output_df is not None and len(existing_output_df) == len(source_df) else source_df.copy()
    for metric in active_metrics:
        if metric.name not in output_df.columns:
            output_df[metric.name] = None
        metric_values = [row.get(metric.name) for row in results]
        for position, target_idx in enumerate(target_indices):
            output_df.at[target_idx, metric.name] = metric_values[position]
    output_df.to_csv(OUTPUT_CSV, index=False)


async def evaluate_all():
    results = load_existing_results()
    pending_tasks = []

    async def run_row(local_idx, row_idx, row_data):
        question = row_data["question"]
        answer = row_data["generated_answer"]
        reference = row_data["ground_truth_answer"]
        contexts = parse_contexts(row_data.get("context"))
        row_result = await evaluate_row(row_idx, question, answer, reference, contexts, results[local_idx])
        return local_idx, row_result

    for local_idx, (row_idx, row) in enumerate(df.iterrows()):
        if all(results[local_idx].get(metric.name) is not None for metric in active_metrics):
            continue

        pending_tasks.append(asyncio.create_task(run_row(local_idx, row_idx, row)))

        if len(pending_tasks) >= args.concurrency:
            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            pending_tasks = list(pending_tasks)
            for task in done:
                local_idx_done, row_result = await task
                results[local_idx_done] = row_result
                persist_results(results)

    while pending_tasks:
        done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        pending_tasks = list(pending_tasks)
        for task in done:
            local_idx_done, row_result = await task
            results[local_idx_done] = row_result
            persist_results(results)

    return results

results = asyncio.run(evaluate_all())

for i, metric in enumerate(active_metrics):
    df[metric.name] = [r[metric.name] for r in results]

persist_results(results)

print(f"Evaluation saved to {OUTPUT_CSV}")
print(f"\nFirst 3 rows with scores:")
preview_columns = ["question"] + [metric.name for metric in active_metrics if metric.name in df.columns]
print(df[preview_columns].head(3))
print(f"\nAverage scores:")
for metric in active_metrics:
    valid_scores = df[metric.name].dropna()
    if not valid_scores.empty:
        print(f"  {metric.name}: {valid_scores.mean():.4f}")
