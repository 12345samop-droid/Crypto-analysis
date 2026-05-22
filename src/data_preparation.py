import pandas as pd
import numpy as np
import os

def prepare_data(input_csv, output_csv):
    print(f"Loading data from {input_csv}...")
    # Re-download the raw file if it was removed
    if not os.path.exists(input_csv):
        print(f"File {input_csv} not found. Ensure it exists.")
        return

    df = pd.read_csv(input_csv, parse_dates=['open_time'])
    df['date'] = df['open_time'].dt.date
    assets = ['ADA', 'BNB', 'BTC', 'DOGE', 'ETH', 'LTC', 'MATIC', 'SOL', 'TRX', 'XRP']

    daily_records = []

    for asset in assets:
        print(f"Processing asset: {asset}")
        asset_cols = ['open_time', 'date', f'{asset}_open', f'{asset}_high', f'{asset}_low', f'{asset}_close', f'{asset}_volume']
        asset_df = df[asset_cols].copy()

        asset_df.rename(columns={
            f'{asset}_open': 'open',
            f'{asset}_high': 'high',
            f'{asset}_low': 'low',
            f'{asset}_close': 'close',
            f'{asset}_volume': 'volume'
        }, inplace=True)

        # Sort chronologically
        asset_df.sort_values('open_time', inplace=True)

        # Group by date
        grouped = asset_df.groupby('date')

        for date, group in grouped:
            if len(group) < 6:
                continue

            open_val = group['open'].iloc[0]
            high_val = group['high'].max()
            low_val = group['low'].min()
            close_val = group['close'].iloc[-1]
            volume_val = group['volume'].sum()

            # Intraday structure
            # Assume 6 candles per day (some might have more, but we use first 3 / last 3)
            mid = len(group) // 2
            first_half = group.iloc[:mid]
            second_half = group.iloc[mid:]

            first_half_return = (first_half['close'].iloc[-1] - first_half['open'].iloc[0]) / first_half['open'].iloc[0] if first_half['open'].iloc[0] != 0 else 0
            second_half_return = (second_half['close'].iloc[-1] - second_half['open'].iloc[0]) / second_half['open'].iloc[0] if second_half['open'].iloc[0] != 0 else 0

            high_idx = group['high'].values.argmax()
            low_idx = group['low'].values.argmin()

            last_candle_return = (group['close'].iloc[-1] - group['open'].iloc[-1]) / group['open'].iloc[-1] if group['open'].iloc[-1] != 0 else 0

            mean_vol = group['volume'].mean()
            volume_spike = group['volume'].max() / mean_vol if mean_vol != 0 else 1

            daily_records.append({
                'date': date,
                'asset': asset,
                'open': open_val,
                'high': high_val,
                'low': low_val,
                'close': close_val,
                'volume': volume_val,
                'candle_count': len(group),
                'first_half_return': first_half_return,
                'second_half_return': second_half_return,
                'high_idx': high_idx,
                'low_idx': low_idx,
                'last_candle_return': last_candle_return,
                'volume_spike': volume_spike
            })

    valid_daily_df = pd.DataFrame(daily_records)

    # Sort by asset and date
    valid_daily_df.sort_values(['asset', 'date'], inplace=True)
    valid_daily_df.reset_index(drop=True, inplace=True)

    valid_daily_df['next_date'] = valid_daily_df.groupby('asset')['date'].shift(-1)
    valid_daily_df['next_open'] = valid_daily_df.groupby('asset')['open'].shift(-1)
    valid_daily_df['next_high'] = valid_daily_df.groupby('asset')['high'].shift(-1)
    valid_daily_df['next_low'] = valid_daily_df.groupby('asset')['low'].shift(-1)

    import datetime
    valid_daily_df['date_plus_1'] = valid_daily_df['date'] + datetime.timedelta(days=1)
    valid_row_mask = valid_daily_df['next_date'] == valid_daily_df['date_plus_1']

    valid_daily_df['distance_up'] = valid_daily_df['next_high'] - valid_daily_df['next_open']
    valid_daily_df['distance_down'] = valid_daily_df['next_open'] - valid_daily_df['next_low']

    conditions = [
        valid_daily_df['distance_up'] > valid_daily_df['distance_down'],
        valid_daily_df['distance_down'] > valid_daily_df['distance_up']
    ]
    choices = [1, 0]
    valid_daily_df['target'] = np.select(conditions, choices, default=-1)
    valid_daily_df['tie_flag'] = (valid_daily_df['target'] == -1).astype(int)

    final_df = valid_daily_df[valid_row_mask].copy()
    final_df.drop(columns=['next_date', 'date_plus_1', 'next_open', 'next_high', 'next_low', 'distance_up', 'distance_down'], inplace=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved daily dataset to {output_csv}")

if __name__ == '__main__':
    prepare_data('data/CRYPTO_MASTER_4H_ALL_ASSETS.csv', 'data/daily_data.csv')
