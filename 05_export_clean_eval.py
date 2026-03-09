import re

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font


INPUT_CSV = "04_evaluated.csv"
OUTPUT_XLSX = "05_evaluated_clean.xlsx"
USED_COLUMNS = [
    "question",
    "ground_truth_answer",
    "generated_answer",
    "faithfulness",
    "answer_relevancy",
]
DISPLAY_NAMES = {
    "question": "question",
    "ground_truth_answer": "ground_truth_answer",
    "generated_answer": "generated_answer",
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "evaluation_status": "status",
}
WIDTH_LIMITS = {
    "A": 42,
    "B": 60,
    "C": 60,
    "D": 16,
    "E": 18,
    "F": 14,
}


def parse_metric_value(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() == "none":
        return None

    match = re.search(r"MetricResult\(value=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)", text)
    if match:
        return float(match.group(1))

    try:
        return float(text)
    except ValueError:
        return None


def adjust_excel_layout(path):
    workbook = load_workbook(path)
    worksheet = workbook["evaluation"]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in worksheet.iter_rows(min_row=2):
        for cell in row:
            if cell.column_letter in {"A", "B", "C"}:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            else:
                cell.alignment = Alignment(vertical="top")

    for column_cells in worksheet.columns:
        column_letter = column_cells[0].column_letter
        max_length = max(len(str(cell.value or "")) for cell in column_cells)
        worksheet.column_dimensions[column_letter].width = min(max_length + 2, WIDTH_LIMITS.get(column_letter, 20))

    workbook.save(path)


df = pd.read_csv(INPUT_CSV)

for column in ["faithfulness", "answer_relevancy"]:
    df[column] = df[column].apply(parse_metric_value)

df = df[USED_COLUMNS].copy()
df = df[df[["faithfulness", "answer_relevancy"]].notna().all(axis=1)].copy()
df["evaluation_status"] = "complete"
df = df.rename(columns=DISPLAY_NAMES)

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="evaluation")

adjust_excel_layout(OUTPUT_XLSX)

print(f"Clean Excel exported to {OUTPUT_XLSX}")
print(f"Rows exported: {len(df)}")
