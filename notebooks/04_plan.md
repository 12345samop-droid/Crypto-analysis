# Research Plan

1. **Setup & Data Preparation (Strict rules)**
   - Create robust daily grouping based strictly on timestamp date in the given timezone.
   - Aggregate 4H data into daily OHLCV. Drop incomplete days at the edges (e.g., if there's only 2 or 4 rows instead of 6) to avoid partial-day artifacts if necessary. Wait, the instructions say: "There are partial edge days in the file; handle them carefully and document what you exclude and why." We should exclude days with fewer than 6 candles to ensure fair daily context, or just document that the first/last days might be partial and exclude them if they have fewer than 6 periods.
   - Define target: `1` if `distance_up > distance_down` else `0`. Ties (-1) will be excluded from the train/test sets to avoid leakage or bias.
   - Save the cleaned daily dataframe.

2. **Feature Engineering (Only past data)**
   - **Hypothesis 1: Daily candle structure from day t**. Features: daily returns, body size relative to range, wick sizes, close location in range (0-1), expansion/compression (range_t / range_t-1).
   - **Hypothesis 2: Multi-day context**. Features: 3-day, 5-day, 10-day moving averages of returns, volatility (ATR/STD), RSI, MACD-like momentum, consecutive up/down days.
   - **Hypothesis 3: Intraday structure from day t**. Features: first half return (first 3 candles) vs second half return (last 3 candles), where the high occurred (which 4H index), where the low occurred.
   - **Hypothesis 4: Cross-asset context**. Features: BTC returns, BTC volatility, BTC close in range.
   - **Hypothesis 5: Regime segmentation**. Features: trending vs ranging (ADX-like), breakout (close > 5-day high).
   - *Ensure no lookahead bias.*

3. **Experiment Setup & Validation Framework**
   - **Time-based split**: Train on 2020 to mid-2023, Test on mid-2023 to 2024.
   - **Walk-forward testing**: Sliding or expanding window over the years to evaluate out-of-sample (OOS) consistency.
   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
   - Stress testing: Check if accuracy > 50% significantly, robust across different assets.

4. **Iterative Model Search**
   - *Iteration 1*: Simple baseline (predict majority class or simple reversal).
   - *Iteration 2*: Simple logic rules (e.g., if close is near the high, predict 0 - mean reversion).
   - *Iteration 3*: Logistic Regression with simple daily features.
   - *Iteration 4*: Random Forest / XGBoost with all features to capture non-linearities. Feature importance check.
   - *Iteration 5*: Walk-forward testing of the best model.
   - *Iteration 6*: Edge survival testing and robustness across all assets.
   - Reject bad models, keep logs of why they failed.

5. **Finalization & Documentation**
   - Document the best performing model or rule.
   - Run the final model on the exact OOS validation set.
   - If no robust edge is found, declare it rejected.
   - Complete pre-commit and push all scripts, logs, results.
