import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

model = os.getenv("LLM_MODEL_NAME")
if not model:
    raise ValueError("LLM_MODEL_NAME not found in .env file")


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
    if not clean_context:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for academic questions. "
                    "Always answer in clear English. "
                    "If no useful context is provided, answer using your best general knowledge and state briefly when you are relying on general knowledge."
                ),
            },
            {"role": "user", "content": f"Question:\n{question}"},
        ]

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


def generate_answer(question, context):
    try:
        response = client.chat.completions.create(
            model=str(model),
            messages=build_prompt(question, context),
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating answer for question: {str(question)[:50]}... Error: {e}")
        return None


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
    df_test = df.iloc[args.start:end_idx].copy()

    print(
        f"Generating answers for {len(df_test)} questions from row {args.start} "
        f"with concurrency={args.concurrency}..."
    )

    generated_answers = [None] * len(df_test)
    rows = list(df_test.iterrows())

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_map = {
            executor.submit(generate_answer, row["question"], row.get("context")): output_idx
            for output_idx, (_, row) in enumerate(rows)
        }

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Generating answers"):
            output_idx = future_map[future]
            generated_answers[output_idx] = future.result()

    df_test["generated_answer"] = generated_answers
    df_test.to_csv(args.output, index=False)

    print(f"Answers saved to {args.output}")


if __name__ == "__main__":
    main()
