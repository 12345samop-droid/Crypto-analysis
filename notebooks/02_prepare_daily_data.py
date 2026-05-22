import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/CRYPTO_MASTER_4H_ALL_ASSETS.csv', parse_dates=['open_time'])

# The requirements say: "Define a 'day' using the timestamp date from open_time in the file's timezone."
df['date'] = df['open_time'].dt.date

print(f"Total rows: {len(df)}")
print(f"Total days: {df['date'].nunique()}")
print(df.groupby('date').size().value_counts().sort_index())
