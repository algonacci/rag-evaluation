import pandas as pd
from dotenv import load_dotenv
import os
import asyncio
import re
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from tqdm import tqdm

load_dotenv()

evaluator_client = AsyncOpenAI(
    base_url=os.getenv('EVALUATOR_LLM_BASE_URL') or os.getenv('EVALUATOR_GENERATOR_LLM_BASE_URL'),
    api_key=os.getenv('EVALUATOR_LLM_API_KEY')
)

evaluator_model = os.getenv('EVALUATOR_LLM_MODEL_NAME')
if not evaluator_model:
    raise ValueError("EVALUATOR_LLM_MODEL_NAME not found in .env file")

evaluator_llm = llm_factory(evaluator_model, client=evaluator_client)

embeddings_client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    client=embeddings_client
)

metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings)
]

df = pd.read_csv('03_with_answers.csv')
OUTPUT_CSV = '04_evaluated.csv'

RETRY_LIMITS = [
    {"question": 400, "answer": 1600, "reference": 1600},
    {"question": 300, "answer": 900, "reference": 900},
    {"question": 220, "answer": 600, "reference": 600},
    {"question": 180, "answer": 450, "reference": 450},
    {"question": 140, "answer": 320, "reference": 320},
    {"question": 120, "answer": 220, "reference": 220},
]

print(f"Evaluating {len(df)} answers...")


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


def build_inputs(question, answer, reference, limits):
    clean_question = compress_text(question, limits["question"], max_sentences=2)
    clean_answer = compress_text(answer, limits["answer"], max_sentences=4)
    clean_reference = compress_text(reference, limits["reference"], max_sentences=4)
    supporting_context = clean_reference or clean_answer
    return clean_question, clean_answer, supporting_context


async def evaluate_metric(metric, question, answer, reference, limits):
    clean_question, clean_answer, supporting_context = build_inputs(question, answer, reference, limits)

    if metric.name == "faithfulness":
        score = await metric.ascore(
            user_input=clean_question,
            response=clean_answer,
            retrieved_contexts=[supporting_context],
        )
    elif metric.name == "answer_relevancy":
        score = await metric.ascore(
            user_input=clean_question,
            response=clean_answer,
        )
    else:
        return None

    return extract_metric_value(score)


def get_metric_retry_limits(metric_name):
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


async def evaluate_row(idx, question, answer, reference, existing_result):
    results = {metric.name: existing_result.get(metric.name) for metric in metrics}

    if pd.isna(answer) or str(answer).strip() == '':
        return results

    for metric in metrics:
        if results.get(metric.name) is not None:
            continue

        last_error = None
        for limits in get_metric_retry_limits(metric.name):
            try:
                results[metric.name] = await evaluate_metric(metric, question, answer, reference, limits)
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
    if not os.path.exists(OUTPUT_CSV):
        return [{} for _ in range(len(df))]

    existing_df = pd.read_csv(OUTPUT_CSV)
    existing_results = []
    for _, row in existing_df.iterrows():
        row_result = {}
        for metric in metrics:
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
    output_df = df.copy()
    for metric in metrics:
        output_df[metric.name] = [row.get(metric.name) for row in results]
    output_df.to_csv(OUTPUT_CSV, index=False)


async def evaluate_all():
    results = load_existing_results()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        if all(results[idx].get(metric.name) is not None for metric in metrics):
            continue

        question = row['question']
        answer = row['generated_answer']
        reference = row['ground_truth_answer']

        results[idx] = await evaluate_row(idx, question, answer, reference, results[idx])
        persist_results(results)

    return results

results = asyncio.run(evaluate_all())

for i, metric in enumerate(metrics):
    df[metric.name] = [r[metric.name] for r in results]

persist_results(results)

print(f"Evaluation saved to {OUTPUT_CSV}")
print(f"\nFirst 3 rows with scores:")
print(df[['question', 'faithfulness', 'answer_relevancy']].head(3))
print(f"\nAverage scores:")
for metric in metrics:
    valid_scores = df[metric.name].dropna()
    if not valid_scores.empty:
        print(f"  {metric.name}: {valid_scores.mean():.4f}")
