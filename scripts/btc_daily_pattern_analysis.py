"""BTC daily pattern analysis (1:30 AM IST daily boundary)

Creates daily OHLC for BTC using 4H candles in data/CRYPTO_MASTER_4H_ALL_ASSETS.csv,
tests several simple, explainable rules (no black-box ML), and reports
Immediate and Immediate+Relaxed (T+2..T+11) win rates.

Usage:
  python3 scripts/btc_daily_pattern_analysis.py

Outputs:
  - prints candidate-rule table to stdout
  - saves results to outputs/btc_pattern_results.json

Notes:
  - Assumes input CSV at data/CRYPTO_MASTER_4H_ALL_ASSETS.csv
  - Uses pandas, numpy
"""

import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np

INPUT_CSV = 'data/CRYPTO_MASTER_4H_ALL_ASSETS.csv'
OUTPUT_JSON = 'outputs/btc_pattern_results.json'

os.makedirs('outputs', exist_ok=True)


def load_and_aggregate(csv_path):
    # Parse timestamps (they include +05:30 tz offset). Force tz-aware and convert to Asia/Kolkata
    df = pd.read_csv(csv_path, parse_dates=['open_time'], low_memory=False)
    # Ensure timezone-aware
    if df['open_time'].dt.tz is None:
        # If naive, assume the timestamps are already with +05:30 in the string; pandas should parse tz.
        df['open_time'] = pd.to_datetime(df['open_time']).dt.tz_localize('Asia/Kolkata', ambiguous='NaT', nonexistent='shift_forward')
    else:
        df['open_time'] = df['open_time'].dt.tz_convert('Asia/Kolkata')

    # Define day bucket that starts at 01:30 IST. For a timestamp t, assign it to day = (t - 1h30m).date()
    df['day'] = (df['open_time'] - pd.Timedelta(hours=1, minutes=30)).dt.date

    # We'll aggregate BTC columns
    btc_cols = ['BTC_open','BTC_high','BTC_low','BTC_close','BTC_volume']
    # Some CSVs may have slightly different names; ensure they exist
    for c in btc_cols:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in CSV")

    # Aggregate per day
    grouped = df.sort_values('open_time').groupby('day')
    rows = []
    for day, g in grouped:
        btc_open = g.iloc[0]['BTC_open']
        btc_close = g.iloc[-1]['BTC_close']
        btc_high = g['BTC_high'].max()
        btc_low = g['BTC_low'].min()
        btc_vol = g['BTC_volume'].sum()

        # other-assets volume (sum of all *_volume minus BTC_volume)
        vol_cols = [c for c in df.columns if c.endswith('_volume')]
        other_vol = g[vol_cols].sum(axis=1).sum() - g['BTC_volume'].sum()

        # compute position of close in range
        rng = btc_high - btc_low
        close_pos = (btc_close - btc_low) / rng if rng > 0 else np.nan

        rows.append({'day': pd.to_datetime(str(day)).date(),
                     'btc_open': float(btc_open),
                     'btc_close': float(btc_close),
                     'btc_high': float(btc_high),
                     'btc_low': float(btc_low),
                     'btc_vol': float(btc_vol),
                     'other_vol': float(other_vol),
                     'close_pos': float(close_pos)})

    daily = pd.DataFrame(rows).sort_values('day').reset_index(drop=True)
    return daily


def backtest_rules(daily):
    # Prepare rolling averages
    daily['vol_14'] = daily['btc_vol'].rolling(14, min_periods=3).mean()
    daily['other_vol_14'] = daily['other_vol'].rolling(14, min_periods=3).mean()

    results = []

    # Define rule functions: each returns prediction for day t+1 given index t
    def rule_momentum(prev):
        # If previous day was green => predict green, else predict red
        return 'Green' if prev['btc_close'] > prev['btc_open'] else 'Red'

    def rule_close_near_high(prev):
        # If previous close in top 20% of range => predict Red (mean reversion)
        if pd.isna(prev['close_pos']):
            return None
        return 'Red' if prev['close_pos'] >= 0.8 else 'Green'

    def rule_volume_confirm(prev):
        # If prev day vol > 1.5 * vol_14 => predict same direction as prev (volume confirms momentum)
        if pd.isna(prev['vol_14']):
            return None
        if prev['btc_vol'] > 1.5 * prev['vol_14']:
            return 'Green' if prev['btc_close'] > prev['btc_open'] else 'Red'
        return None

    def rule_altcoin_volume(prev):
        # If other assets had a >1.8x volume spike and prev btc green => predict Green
        if pd.isna(prev['other_vol_14']):
            return None
        if prev['other_vol'] > 1.8 * prev['other_vol_14'] and prev['btc_close'] > prev['btc_open']:
            return 'Green'
        return None

    candidate_rules = [
        ('Momentum(prev day)', rule_momentum),
        ('CloseNearHigh(prev day) => Red if close_pos>=0.8', rule_close_near_high),
        ('VolumeConfirm(prev day) if vol>1.5*avg14', rule_volume_confirm),
        ('AltcoinVolumeSpike(prev day) && prev green => Green', rule_altcoin_volume),
    ]

    # For each rule, iterate over days and make prediction for day t+1 using data up to day t
    N = len(daily)
    for name, func in candidate_rules:
        stats = {'name': name, 'total_signals': 0, 'immediate_wins': 0, 'relaxed_wins': 0}
        # track per-signal details optionally
        for t in range(0, N-1):
            prev = daily.loc[t]
            pred = func(prev)
            if pred is None:
                continue
            stats['total_signals'] += 1
            # Evaluate on target day = t+1
            target = daily.loc[t+1]
            actual = 'Green' if target['btc_close'] > target['btc_open'] else 'Red' if target['btc_close'] < target['btc_open'] else 'Flat'
            if actual == pred:
                stats['immediate_wins'] += 1
                stats['relaxed_wins'] += 1
            else:
                # check relaxed recovery window T+2..T+11 (indices t+2 .. t+11)
                open_t1 = target['btc_open']
                recovered = False
                # If predicted Green but actual Red, recovery = any high in window >= open_t1
                # If predicted Red but actual Green, recovery = any low in window <= open_t1
                for w in range(t+2, min(t+12, N)):
                    dayw = daily.loc[w]
                    if pred == 'Green' and dayw['btc_high'] >= open_t1:
                        recovered = True
                        break
                    if pred == 'Red' and dayw['btc_low'] <= open_t1:
                        recovered = True
                        break
                if recovered:
                    stats['relaxed_wins'] += 1
        # compute rates
        tot = stats['total_signals'] if stats['total_signals']>0 else 1
        stats['immediate_rate'] = stats['immediate_wins'] / tot
        stats['relaxed_rate'] = stats['relaxed_wins'] / tot
        results.append(stats)

    return results


def find_best_rule(results):
    # Choose rule with highest relaxed_rate (or immediate_rate preferentially)
    df = pd.DataFrame(results)
    # Rank by relaxed_rate then immediate_rate
    df['rank_score'] = df['relaxed_rate']*0.6 + df['immediate_rate']*0.4
    best = df.sort_values(['rank_score'], ascending=False).iloc[0].to_dict()
    return best, df.sort_values('rank_score', ascending=False).to_dict(orient='records')


def main():
    print('Loading and aggregating CSV into daily BTC...')
    daily = load_and_aggregate(INPUT_CSV)
    print(f'Loaded {len(daily)} daily rows')

    print('Backtesting candidate rules...')
    results = backtest_rules(daily)

    best, ranked = find_best_rule(results)

    out = {'summary': best, 'ranked': ranked}
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    print('\nTop candidate rule:')
    print(json.dumps(best, indent=2))
    print(f"Full results saved to {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
