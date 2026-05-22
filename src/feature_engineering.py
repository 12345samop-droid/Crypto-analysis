import pandas as pd
import numpy as np
import os

def create_features(input_csv, output_csv):
    print(f"Loading daily data from {input_csv}...")
    df = pd.read_csv(input_csv, parse_dates=['date'])

    df.sort_values(by=['asset', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    asset_dfs = []

    for asset, asset_df in df.groupby('asset'):
        asset_df = asset_df.copy()

        # A. Candle structure
        asset_df['f_body_size'] = abs(asset_df['close'] - asset_df['open'])
        asset_df['f_range'] = asset_df['high'] - asset_df['low']
        safe_range = asset_df['f_range'].replace(0, 1e-8)

        asset_df['f_body_range_ratio'] = asset_df['f_body_size'] / safe_range
        asset_df['f_upper_wick'] = asset_df['high'] - asset_df[['open', 'close']].max(axis=1)
        asset_df['f_lower_wick'] = asset_df[['open', 'close']].min(axis=1) - asset_df['low']
        asset_df['f_upper_wick_ratio'] = asset_df['f_upper_wick'] / safe_range
        asset_df['f_lower_wick_ratio'] = asset_df['f_lower_wick'] / safe_range
        asset_df['f_close_location'] = (asset_df['close'] - asset_df['low']) / safe_range
        asset_df['f_direction'] = np.where(asset_df['close'] >= asset_df['open'], 1, -1)

        asset_df['f_prev_range'] = asset_df['f_range'].shift(1)
        asset_df['f_range_expansion'] = asset_df['f_range'] / asset_df['f_prev_range'].replace(0, 1e-8)

        # B. Multi-day context
        asset_df['f_return'] = asset_df['close'].pct_change(1)

        for w in [2, 3, 5, 10, 20]:
            asset_df[f'f_roll_return_mean_{w}'] = asset_df['f_return'].rolling(window=w).mean()
            asset_df[f'f_roll_return_std_{w}'] = asset_df['f_return'].rolling(window=w).std()

            roll_high = asset_df['high'].rolling(window=w).max()
            roll_low = asset_df['low'].rolling(window=w).min()

            asset_df[f'f_dist_to_high_{w}'] = (roll_high - asset_df['close']) / asset_df['close']
            asset_df[f'f_dist_to_low_{w}'] = (asset_df['close'] - roll_low) / asset_df['close']

            asset_df[f'f_avg_range_{w}'] = asset_df['f_range'].rolling(window=w).mean()
            asset_df[f'f_vol_regime_{w}'] = asset_df['f_range'] / asset_df[f'f_avg_range_{w}'].replace(0, 1e-8)

        up_day = (asset_df['f_return'] > 0).astype(int)
        down_day = (asset_df['f_return'] < 0).astype(int)

        def get_consecutive(s):
            return s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)

        asset_df['f_cons_up'] = get_consecutive(up_day)
        asset_df['f_cons_down'] = get_consecutive(down_day)

        # C. Intraday Structure
        asset_df['f_first_half_return'] = asset_df['first_half_return']
        asset_df['f_second_half_return'] = asset_df['second_half_return']
        asset_df['f_high_idx'] = asset_df['high_idx']
        asset_df['f_low_idx'] = asset_df['low_idx']
        asset_df['f_last_candle_return'] = asset_df['last_candle_return']
        asset_df['f_volume_spike'] = asset_df['volume_spike']

        # Sequence logic: did high happen before low?
        asset_df['f_high_before_low'] = (asset_df['high_idx'] < asset_df['low_idx']).astype(int)

        # E. Regime features
        asset_df['f_trend_strength'] = abs(asset_df['close'].diff(10)) / (asset_df['f_range'].rolling(10).sum().replace(0, 1e-8))
        asset_df['f_exhaustion'] = (asset_df['f_body_size'] > asset_df['f_avg_range_10'] * 1.5).astype(int) * asset_df['f_direction']

        asset_dfs.append(asset_df)

    combined_df = pd.concat(asset_dfs, ignore_index=True)
    combined_df.sort_values(by=['date', 'asset'], inplace=True)

    # D. Cross-asset context
    btc_df = combined_df[combined_df['asset'] == 'BTC'][['date', 'f_return', 'f_vol_regime_10']].copy()
    btc_df.rename(columns={
        'f_return': 'f_btc_return',
        'f_vol_regime_10': 'f_btc_vol_regime_10'
    }, inplace=True)

    combined_df = combined_df.merge(btc_df, on='date', how='left')

    # F. Calendar features
    combined_df['f_weekday'] = combined_df['date'].dt.weekday
    combined_df['f_month'] = combined_df['date'].dt.month
    combined_df['f_quarter'] = combined_df['date'].dt.quarter

    feature_cols = [c for c in combined_df.columns if c.startswith('f_')]
    meta_cols = ['date', 'asset', 'target', 'tie_flag']

    final_features_df = combined_df[meta_cols + feature_cols].dropna().copy()

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_features_df.to_csv(output_csv, index=False)
    print(f"Saved feature dataset to {output_csv}")

if __name__ == '__main__':
    create_features('data/daily_data.csv', 'data/features.csv')
