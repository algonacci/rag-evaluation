import pandas as pd

df = pd.read_excel('data_awal.xlsx', skiprows=1)
print(df.to_string())
print(f"\nTotal rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
