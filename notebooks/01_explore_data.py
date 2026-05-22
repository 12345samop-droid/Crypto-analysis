import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/CRYPTO_MASTER_4H_ALL_ASSETS.csv', parse_dates=['open_time'])
print(df.info())
print(df.head())
print(df.tail())

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Assets
assets = ['ADA', 'BNB', 'BTC', 'DOGE', 'ETH', 'LTC', 'MATIC', 'SOL', 'TRX', 'XRP']
for asset in assets:
    cols = [c for c in df.columns if c.startswith(asset)]
    print(f"Asset: {asset}, Columns: {cols}")
