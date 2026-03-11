import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge context into existing answers without regenerating.")
    parser.add_argument("--base", default="02_data_evaluation.csv", help="Base CSV with context")
    parser.add_argument("--answers", default="03_with_answers.csv", help="CSV with generated answers")
    parser.add_argument("--output", default="03_with_answers.csv", help="Output CSV path")
    args = parser.parse_args()

    base_df = pd.read_csv(args.base)
    ans_df = pd.read_csv(args.answers)

    if "question" not in base_df.columns:
        raise ValueError("Base CSV must contain 'question' column.")
    if "question" not in ans_df.columns or "generated_answer" not in ans_df.columns:
        raise ValueError("Answers CSV must contain 'question' and 'generated_answer'.")

    ans_df = ans_df.drop_duplicates(subset=["question"])
    merged = base_df.merge(ans_df[["question", "generated_answer"]], on="question", how="left", suffixes=("", "_old"))

    if "generated_answer_old" in merged.columns:
        merged["generated_answer"] = merged["generated_answer"].fillna(merged["generated_answer_old"])
        merged = merged.drop(columns=["generated_answer_old"])

    merged.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
