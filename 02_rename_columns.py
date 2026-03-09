import pandas as pd

df = pd.read_excel('data_awal.xlsx', skiprows=1)

column_mapping = {
    'Question': 'question',
    'Pertanyaan': 'question',
    'jawaban': 'ground_truth_answer',
    'Context': 'context',
    'Precision': 'context_precision',
    'Faithfullness': 'faithfulness',
    'Recall': 'context_recall'
}

df = df.rename(columns=column_mapping)

required_columns = [
    'question',
    'ground_truth_answer',
    'generated_answer',
    'context_precision',
    'context_recall',
    'context_entities_recall',
    'noise_sensitivity',
    'response_relevancy',
    'faithfulness'
]

for col in required_columns:
    if col not in df.columns:
        df[col] = None

df = df[required_columns]

empty_gt = df[df['ground_truth_answer'].isna() | (df['ground_truth_answer'].astype(str).str.strip() == '')]

if not empty_gt.empty:
    print("Rows with EMPTY ground truth answer:")
    for idx, row in empty_gt.iterrows():
        print(f"  Row {idx}: {row['question'][:80]}...")

df = df[df['ground_truth_answer'].notna() & (df['ground_truth_answer'].astype(str).str.strip() != '')]

df.to_csv('02_data_evaluation.csv', index=False)

print(f"\nFile converted to 02_data_evaluation.csv")
print(f"Total rows (after filtering): {len(df)}")
print(f"Filtered out: {len(empty_gt)} rows")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))
