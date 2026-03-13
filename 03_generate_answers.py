import argparse
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "120")),
)

model = os.getenv("LLM_MODEL_NAME")
if not model:
    raise ValueError("LLM_MODEL_NAME not found in .env file")

GENERATION_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
GENERATION_TOP_P = float(os.getenv("LLM_TOP_P", "1"))
GENERATION_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
GENERATION_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
GENERATION_CONTEXT_MAX_CHARS = int(os.getenv("LLM_CONTEXT_MAX_CHARS", "3500"))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate answers using question + retrieved context.")
    parser.add_argument("--input", default="02_data_evaluation.csv", help="Input CSV path.")
    parser.add_argument("--output", default="03_with_answers.csv", help="Output CSV path.")
    parser.add_argument("--start", type=int, default=0, help="Start row index (0-based).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to process.")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel requests to send.")
    return parser.parse_args()


def build_prompt(question, context):
    clean_context = "" if pd.isna(context) else str(context).strip()
    if GENERATION_CONTEXT_MAX_CHARS <= 0:
        clean_context = ""
    elif clean_context and len(clean_context) > GENERATION_CONTEXT_MAX_CHARS:
        clean_context = clean_context[: GENERATION_CONTEXT_MAX_CHARS - 3].rstrip() + "..."
    if not clean_context:
        return build_fallback_prompt(question)

    return [
        {
            "role": "system",
            "content": (
                "You are a retrieval-augmented assistant for academic papers. "
                "Prioritize the provided context when it is relevant and sufficient. "
                "If the retrieved context is incomplete, noisy, or not relevant enough, you may fall back to your best general knowledge. "
                "When you do so, say briefly that you are relying partly on general knowledge. "
                "Always answer in clear English."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved context:\n{clean_context}\n\n"
                "Provide a concise answer. Use the retrieved context first, but do not refuse to answer just because the context is weak."
            ),
        },
    ]


def build_fallback_prompt(question):
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for academic questions. "
                "Always answer in clear English. "
                "If no useful context is available, answer using your best general knowledge. "
                "Do not refuse to answer."
            ),
        },
        {"role": "user", "content": f"Question:\n{question}"},
    ]


def generate_answer(question, context):
    try:
        response = client.chat.completions.create(
            model=str(model),
            messages=build_prompt(question, context),
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            max_tokens=GENERATION_MAX_TOKENS,
            timeout=GENERATION_TIMEOUT,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Primary generation failed for question: {str(question)[:50]}... Error: {e}")
        try:
            response = client.chat.completions.create(
                model=str(model),
                messages=build_fallback_prompt(question),
                temperature=GENERATION_TEMPERATURE,
                top_p=GENERATION_TOP_P,
                max_tokens=GENERATION_MAX_TOKENS,
                timeout=GENERATION_TIMEOUT,
            )
            return response.choices[0].message.content
        except Exception as fallback_error:
            print(
                "Fallback generation failed for question: "
                f"{str(question)[:50]}... Error: {fallback_error}"
            )
            return None


def generate_answer_with_alarm(question, context):
    if GENERATION_TIMEOUT <= 0:
        return generate_answer(question, context)

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Generation timed out after {GENERATION_TIMEOUT} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, GENERATION_TIMEOUT)
    try:
        return generate_answer(question, context)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")

    end_idx = None if args.limit is None else args.start + args.limit
    target_indices = list(df.iloc[args.start:end_idx].index)

    if "generated_answer" not in df.columns:
        df["generated_answer"] = None
    df["generated_answer"] = df["generated_answer"].astype(object)

    if os.path.exists(args.output):
        existing_df = pd.read_csv(args.output)
        if len(existing_df) == len(df) and "generated_answer" in existing_df.columns:
            df["generated_answer"] = existing_df["generated_answer"].astype(object)
        elif len(existing_df) == len(target_indices) and "generated_answer" in existing_df.columns:
            df.loc[target_indices, "generated_answer"] = existing_df["generated_answer"].astype(object).values

    df_test = df.loc[target_indices].copy()

    write_lock = threading.Lock()

    def persist_progress():
        with write_lock:
            df.to_csv(args.output, index=False)

    print(
        f"Generating answers for {len(df_test)} questions from row {args.start} "
        f"with concurrency={args.concurrency}, temperature={GENERATION_TEMPERATURE}, "
        f"top_p={GENERATION_TOP_P}, max_tokens={GENERATION_MAX_TOKENS}, timeout={GENERATION_TIMEOUT}s, "
        f"context_max_chars={GENERATION_CONTEXT_MAX_CHARS}..."
    )

    rows = list(df_test.iterrows())
    pending_rows = [
        (output_idx, row)
        for output_idx, (_, row) in enumerate(rows)
        if pd.isna(df_test.iloc[output_idx].get("generated_answer"))
        or not str(df_test.iloc[output_idx].get("generated_answer")).strip()
    ]

    if not pending_rows:
        print(f"Answers already present in {args.output}")
        return

    persist_progress()

    if args.concurrency == 1:
        for output_idx, row in tqdm(pending_rows, total=len(pending_rows), desc="Generating answers"):
            target_idx = df_test.index[output_idx]
            df.at[target_idx, "generated_answer"] = generate_answer_with_alarm(
                row["question"], row.get("context")
            )
            persist_progress()
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_map = {
                executor.submit(generate_answer, row["question"], row.get("context")): output_idx
                for output_idx, row in pending_rows
            }

            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Generating answers"):
                output_idx = future_map[future]
                target_idx = df_test.index[output_idx]
                df.at[target_idx, "generated_answer"] = future.result()
                persist_progress()

    print(f"Answers saved to {args.output}")


if __name__ == "__main__":
    main()
