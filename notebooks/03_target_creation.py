import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/CRYPTO_MASTER_4H_ALL_ASSETS.csv', parse_dates=['open_time'])
df['date'] = df['open_time'].dt.date

assets = ['ADA', 'BNB', 'BTC', 'DOGE', 'ETH', 'LTC', 'MATIC', 'SOL', 'TRX', 'XRP']

daily_data = []

# Group by date to get daily OHLC
# We have to be careful not to use any information from the future.
for date, group in df.groupby('date'):
    # Sort by open_time
    group = group.sort_values('open_time')

    daily_row = {'date': date}

    for asset in assets:
        open_col = f'{asset}_open'
        high_col = f'{asset}_high'
        low_col = f'{asset}_low'
        close_col = f'{asset}_close'
        vol_col = f'{asset}_volume'

        daily_row[f'{asset}_daily_open'] = group[open_col].iloc[0]
        daily_row[f'{asset}_daily_high'] = group[high_col].max()
        daily_row[f'{asset}_daily_low'] = group[low_col].min()
        daily_row[f'{asset}_daily_close'] = group[close_col].iloc[-1]
        daily_row[f'{asset}_daily_volume'] = group[vol_col].sum()

    daily_data.append(daily_row)

df_daily = pd.DataFrame(daily_data)
df_daily = df_daily.sort_values('date').reset_index(drop=True)

# Now define target for day t+1
# For each asset and for each valid day t, use only information available up to the end of day t to predict, for day t+1, which extreme will be farther from t+1 open:
# - distance_up = next_day_high - next_day_open
# - distance_down = next_day_open - next_day_low
# Define the binary target:
# - 1 if distance_up > distance_down
# - 0 if distance_down > distance_up
# Handle ties explicitly and consistently. Either exclude ties from the main classification or label them separately, but do not mix them into either class without reporting it.

for asset in assets:
    # Get next day's open, high, low
    next_open = df_daily[f'{asset}_daily_open'].shift(-1)
    next_high = df_daily[f'{asset}_daily_high'].shift(-1)
    next_low = df_daily[f'{asset}_daily_low'].shift(-1)

    distance_up = next_high - next_open
    distance_down = next_open - next_low

    # Define target
    # 1 if distance_up > distance_down
    # 0 if distance_down > distance_up
    # -1 for ties
    target = np.where(distance_up > distance_down, 1,
             np.where(distance_down > distance_up, 0, -1))

    df_daily[f'{asset}_target'] = target

print(df_daily[[f'{assets[0]}_target', f'{assets[1]}_target']].value_counts())
print("Ties for ADA:", (df_daily['ADA_target'] == -1).sum())

# Exclude the last row as it won't have a target
df_daily = df_daily.iloc[:-1]
