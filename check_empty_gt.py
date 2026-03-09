import pandas as pd

df = pd.read_excel('data_awal.xlsx', skiprows=1)

print(f"Total rows: {len(df)}")

empty_gt = df[df['jawaban'].isna() | (df['jawaban'].astype(str).str.strip() == '')]

print(f"\nRows with empty ground truth (jawaban): {len(empty_gt)}")

if not empty_gt.empty:
    print("\nQuestions without ground truth:")
    for idx, row in empty_gt.iterrows():
        print(f"\n{idx + 1}. {row['Pertanyaan']}")

    empty_gt[['Pertanyaan']].to_csv('04_no_ground_truth.csv', index=False)
    print(f"\nSaved to 04_no_ground_truth.csv")
