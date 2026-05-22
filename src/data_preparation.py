import pandas as pd
import numpy as np
import os

def prepare_data(input_csv, output_csv):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv, parse_dates=['open_time'])

    # Define day boundaries using timestamp date only
    # Note: open_time is in UTC+05:30. The date component will be used as-is.
    df['date'] = df['open_time'].dt.date

    # Assets
    assets = ['ADA', 'BNB', 'BTC', 'DOGE', 'ETH', 'LTC', 'MATIC', 'SOL', 'TRX', 'XRP']

    daily_records = []

    for asset in assets:
        print(f"Processing asset: {asset}")
        # Extract asset columns
        asset_cols = ['open_time', 'date', f'{asset}_open', f'{asset}_high', f'{asset}_low', f'{asset}_close', f'{asset}_volume']
        asset_df = df[asset_cols].copy()

        # Rename columns to standard names
        asset_df.rename(columns={
            f'{asset}_open': 'open',
            f'{asset}_high': 'high',
            f'{asset}_low': 'low',
            f'{asset}_close': 'close',
            f'{asset}_volume': 'volume'
        }, inplace=True)

        # Group by date to compute daily OHLCV
        grouped = asset_df.groupby('date')

        daily_asset_df = grouped.agg(
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum'),
            candle_count=('open_time', 'count')
        ).reset_index()

        # Add asset column
        daily_asset_df['asset'] = asset

        daily_records.append(daily_asset_df)

    all_daily_df = pd.concat(daily_records, ignore_index=True)

    # Filter out days with candle_count < 6
    initial_count = len(all_daily_df)
    valid_daily_df = all_daily_df[all_daily_df['candle_count'] >= 6].copy()
    filtered_count = initial_count - len(valid_daily_df)
    print(f"Filtered out {filtered_count} days with candle_count < 6. Remaining: {len(valid_daily_df)}")

    # Sort by asset and date
    valid_daily_df.sort_values(['asset', 'date'], inplace=True)
    valid_daily_df.reset_index(drop=True, inplace=True)

    # Compute next day's values for target
    # We must be careful to ensure that the "next row" is actually the next calendar day
    # and that both are valid days.

    valid_daily_df['next_date'] = valid_daily_df.groupby('asset')['date'].shift(-1)
    valid_daily_df['next_open'] = valid_daily_df.groupby('asset')['open'].shift(-1)
    valid_daily_df['next_high'] = valid_daily_df.groupby('asset')['high'].shift(-1)
    valid_daily_df['next_low'] = valid_daily_df.groupby('asset')['low'].shift(-1)

    # Verify that the next row is actually the very next day, if needed.
    # But wait, the instructions say:
    # "Drop any row whose next day is invalid or partial"
    # By grouping and shifting, we get the next row in valid_daily_df.
    # If the next actual calendar day is missing or was filtered out, then next_date will not be date + 1 day.
    # Let's enforce that next_date == date + timedelta(days=1)

    import datetime
    valid_daily_df['date_plus_1'] = valid_daily_df['date'] + datetime.timedelta(days=1)
    valid_row_mask = valid_daily_df['next_date'] == valid_daily_df['date_plus_1']

    # Only compute target for rows where valid_row_mask is True
    valid_daily_df['distance_up'] = valid_daily_df['next_high'] - valid_daily_df['next_open']
    valid_daily_df['distance_down'] = valid_daily_df['next_open'] - valid_daily_df['next_low']

    # Target = 1 if distance_up > distance_down, else 0
    # ties = -1
    conditions = [
        valid_daily_df['distance_up'] > valid_daily_df['distance_down'],
        valid_daily_df['distance_down'] > valid_daily_df['distance_up']
    ]
    choices = [1, 0]
    valid_daily_df['target'] = np.select(conditions, choices, default=-1)

    valid_daily_df['tie_flag'] = (valid_daily_df['target'] == -1).astype(int)

    # Drop invalid next day rows
    print(f"Dropping {(~valid_row_mask).sum()} rows where next day is missing or partial.")
    final_df = valid_daily_df[valid_row_mask].copy()

    # Drop columns that are no longer needed
    final_df.drop(columns=['next_date', 'date_plus_1', 'next_open', 'next_high', 'next_low', 'distance_up', 'distance_down'], inplace=True)

    print(f"Final dataset has {len(final_df)} rows. Tied rows: {final_df['tie_flag'].sum()}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved daily dataset to {output_csv}")

if __name__ == '__main__':
    prepare_data('data/CRYPTO_MASTER_4H_ALL_ASSETS.csv', 'data/daily_data.csv')
