"""scripts/discover_patterns.py

Backtest human-readable rules on BTC daily OHLC derived from
data/CRYPTO_MASTER_4H_ALL_ASSETS.csv using the user-specified
1:30 AM IST trading-day boundary and relaxed-recovery window (T+2..T+11).

Usage:
  python3 scripts/discover_patterns.py --csv data/CRYPTO_MASTER_4H_ALL_ASSETS.csv \
      --output outputs/rule_summary.csv --logs outputs/rule_logs

Outputs:
 - outputs/rule_summary.csv : aggregated metric per tested rule
 - outputs/<rule_name>_log.csv : per-signal rows for each rule

This script tests a set of explicit, human-readable rules and a small
parameter grid for threshold-based rules. It computes Immediate Success
and Relaxed Success (recovery) exactly as defined in the prompt.

NOTE: This is not a black-box ML approach; all rules are deterministic
and fully specified so you can manually verify any trade on a chart.

"""

import argparse
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import pytz
import itertools
import json


IST = pytz.timezone("Asia/Kolkata")


def load_and_aggregate(csv_path):
    # Read CSV; only load BTC columns and open_time to reduce memory
    usecols = None
    # we'll let pandas infer columns; then pick BTC_* cols
    df = pd.read_csv(csv_path, parse_dates=["open_time"], infer_datetime_format=True)

    # Ensure open_time is timezone-aware. The file appears to include offsets (+05:30)
    if df["open_time"].dt.tz is None:
        # assume naive datetimes are UTC
        df["open_time"] = df["open_time"].dt.tz_localize('UTC')

    # Convert to IST and apply the 1:30 IST rollover rule
    df["open_time_ist"] = df["open_time"].dt.tz_convert(IST)

    # trading_day := date( open_time_ist - 1 hour 30 minutes )
    shift = pd.Timedelta(hours=1, minutes=30)
    df["trading_day"] = (df["open_time_ist"] - shift).dt.date

    # Select BTC columns
    btc_cols = [c for c in df.columns if c.startswith("BTC_")]
    expected = ["BTC_open","BTC_high","BTC_low","BTC_close","BTC_volume"]
    for col in expected:
        if col not in btc_cols:
            raise RuntimeError(f"Expected column {col} not found in CSV")

    # Group by trading_day and aggregate
    def first_open(x):
        return x.iloc[0]
    def last_close(x):
        return x.iloc[-1]

    gb = df.sort_values("open_time_ist").groupby("trading_day")

    day_open = gb["BTC_open"].first()
    day_close = gb["BTC_close"].last()
    day_high = gb["BTC_high"].max()
    day_low = gb["BTC_low"].min()
    day_volume = gb["BTC_volume"].sum()

    daily = pd.DataFrame({
        "date": pd.to_datetime(day_open.index),
        "open": day_open.values,
        "high": day_high.values,
        "low": day_low.values,
        "close": day_close.values,
        "volume": day_volume.values,
    })

    daily = daily.sort_values("date").reset_index(drop=True)

    # Features
    daily["close_pos"] = (daily["close"] - daily["low"]) / (daily["high"] - daily["low"]).replace(0, np.nan)
    daily["body"] = daily["close"] - daily["open"]
    daily["body_pct"] = daily["body"] / daily["open"]
    daily["range"] = daily["high"] - daily["low"]
    daily["range_pct"] = daily["range"] / daily["open"]

    # shift next-day open/close for scoring convenience
    daily["next_open"] = daily["open"].shift(-1)
    daily["next_close"] = daily["close"].shift(-1)
    daily["next_date"] = daily["date"].shift(-1)

    return daily


def immediate_label(row):
    if pd.isna(row["next_open"]) or pd.isna(row["next_close"]):
        return None
    if row["next_close"] > row["next_open"]:
        return "G"
    if row["next_close"] < row["next_open"]:
        return "R"
    return "U"


def check_relaxed_recovery(daily, idx, open_t1):
    # For an incorrect prediction on index idx (which refers to day T),
    # we check days idx+2 .. idx+11 inclusive for low <= open_t1 <= high.
    # Note: idx+1 is T+1; idx+2 is T+2.
    start = idx + 2
    end = idx + 11
    if start >= len(daily):
        return False
    subset = daily.loc[start: min(end, len(daily)-1)]
    # If any day's intraday range contains open_t1
    hits = ((subset["low"] <= open_t1) & (subset["high"] >= open_t1)).any()
    return bool(hits)


def evaluate_rule(daily, rule_fn, rule_name, min_signals=1, save_logs=True, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for idx, row in daily.iterrows():
        signal = rule_fn(daily, idx, row)
        if signal is None:
            continue
        # Only consider if next day exists
        if pd.isna(row.get("next_open")):
            continue
        actual = immediate_label(row)
        immediate_success = (actual == signal)
        relaxed_success = False
        if immediate_success:
            relaxed_success = True
        else:
            # check recovery: open price of T+1
            open_t1 = row["next_open"]
            relaxed_success = check_relaxed_recovery(daily, idx, open_t1)

        records.append({
            "t_date": row["date"].date(),
            "signal": signal,
            "t_open": row["open"],
            "t_high": row["high"],
            "t_low": row["low"],
            "t_close": row["close"],
            "t_next_date": row["next_date"].date() if not pd.isna(row.get("next_date")) else None,
            "t1_open": row["next_open"],
            "t1_close": row["next_close"],
            "actual": actual,
            "immediate_success": immediate_success,
            "relaxed_success": relaxed_success,
        })

    if len(records) < min_signals:
        summary = {
            "rule": rule_name,
            "signals": len(records),
            "immediate_wins": 0,
            "immediate_win_rate": None,
            "relaxed_wins": 0,
            "relaxed_win_rate": None,
        }
        if save_logs:
            pd.DataFrame(records).to_csv(os.path.join(out_dir, f"{rule_name}_log.csv"), index=False)
        return summary

    df_logs = pd.DataFrame(records)
    immediate_wins = df_logs["immediate_success"].sum()
    relaxed_wins = df_logs["relaxed_success"].sum()
    immediate_rate = immediate_wins / len(df_logs)
    relaxed_rate = relaxed_wins / len(df_logs)

    if save_logs:
        df_logs.to_csv(os.path.join(out_dir, f"{rule_name}_log.csv"), index=False)

    summary = {
        "rule": rule_name,
        "signals": len(df_logs),
        "immediate_wins": int(immediate_wins),
        "immediate_win_rate": float(immediate_rate),
        "relaxed_wins": int(relaxed_wins),
        "relaxed_win_rate": float(relaxed_rate),
    }
    return summary


# Rule implementations (deterministic, use only data up to and including T)

def rule_momentum(daily, idx, row):
    # If Day T is Green predict Green, else Red. Fires every day except last.
    if pd.isna(row["close"]) or pd.isna(row["open"]):
        return None
    if row["close"] > row["open"]:
        return "G"
    if row["close"] < row["open"]:
        return "R"
    return None


def rule_close_pos_threshold(thresh, side_when_missing='R'):
    def _fn(daily, idx, row):
        cp = row.get("close_pos")
        if pd.isna(cp):
            return side_when_missing
        return "G" if cp >= thresh else "R"
    _fn.__name__ = f"close_pos_ge_{thresh}"
    return _fn


def rule_body_pct_threshold(pos_thr=0.01, neg_thr=-0.01):
    def _fn(daily, idx, row):
        bp = row.get("body_pct")
        if pd.isna(bp):
            return None
        if bp >= pos_thr:
            return "G"
        if bp <= neg_thr:
            return "R"
        return None
    _fn.__name__ = f"body_pct_{neg_thr}_to_{pos_thr}"
    return _fn


def rule_large_range_closepos(range_thr=0.02, closepos_high=0.6, closepos_low=0.4):
    def _fn(daily, idx, row):
        if pd.isna(row.get("range_pct")) or pd.isna(row.get("close_pos")):
            return None
        if row["range_pct"] >= range_thr and row["close_pos"] >= closepos_high:
            return "G"
        if row["range_pct"] >= range_thr and row["close_pos"] <= closepos_low:
            return "R"
        return None
    _fn.__name__ = f"large_range_{range_thr}_cp_{closepos_low}_{closepos_high}"
    return _fn


def rule_two_day_momentum(daily, idx, row):
    # Fires only when previous day exists
    if idx == 0:
        return None
    prev = daily.iloc[idx-1]
    dir_t = np.sign(row["close"] - row["open"]) if not pd.isna(row["close"]) and not pd.isna(row["open"]) else 0
    dir_prev = np.sign(prev["close"] - prev["open"]) if not pd.isna(prev["close"]) and not pd.isna(prev["open"]) else 0
    if dir_t == 0 or dir_prev == 0:
        return None
    if dir_t == dir_prev:
        return "G" if dir_t > 0 else "R"
    return None


def rule_doji_reversal(small_body_mult=0.1, cp_high=0.60, cp_low=0.40):
    def _fn(daily, idx, row):
        if pd.isna(row.get("range")) or row.get("range") == 0:
            return None
        if pd.isna(row.get("body")):
            return None
        if abs(row["body"]) <= small_body_mult * row["range"]:
            cp = row.get("close_pos")
            if pd.isna(cp):
                return None
            if cp >= cp_high:
                return "R"
            if cp <= cp_low:
                return "G"
        return None
    _fn.__name__ = f"doji_rev_bmult_{small_body_mult}_cp_{cp_low}_{cp_high}"
    return _fn


def main(args):
    daily = load_and_aggregate(args.csv)
    print(f"Loaded {len(daily)} aggregated days")

    # Candidate rules
    rules = []
    rules.append(("momentum", rule_momentum))
    rules.append(("two_day_momentum", rule_two_day_momentum))
    # grid for close_pos thresholds
    for thresh in [0.6, 0.65, 0.7, 0.75, 0.8]:
        rules.append((f"close_pos_ge_{thresh}", rule_close_pos_threshold(thresh)))
    # body pct thresholds
    rules.append(("body_pct_1pct", rule_body_pct_threshold(0.01, -0.01)))
    rules.append(("body_pct_0.5pct", rule_body_pct_threshold(0.005, -0.005)))
    # large range
    for rthr in [0.01, 0.02, 0.03]:
        rules.append((f"large_range_{rthr}_cp_0.6_0.4", rule_large_range_closepos(rthr, 0.6, 0.4)))
    # doji reversals
    for sm in [0.08, 0.1, 0.12]:
        rules.append((f"doji_rev_{sm}", rule_doji_reversal(sm, 0.6, 0.4)))

    summaries = []
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    for name, fn in rules:
        print(f"Evaluating rule {name}")
        s = evaluate_rule(daily, fn, name, save_logs=True, out_dir=out_dir)
        summaries.append(s)

    summary_df = pd.DataFrame(summaries).sort_values(by=["relaxed_win_rate"], ascending=False)
    summary_df.to_csv(os.path.join(out_dir, "rule_summary.csv"), index=False)

    # Print top results
    print("Top candidate rules by relaxed_win_rate:")
    print(summary_df.head(20).to_string(index=False))

    # Save daily aggregated for manual inspection
    daily.to_csv(os.path.join(out_dir, "btc_daily_aggregated.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/CRYPTO_MASTER_4H_ALL_ASSETS.csv")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    main(args)
