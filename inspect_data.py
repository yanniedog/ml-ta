import pandas as pd

df = pd.read_parquet('c:/code/ml-ta/data/gold/SOLUSDT_1m_gold.parquet')
print(df.columns.tolist())
