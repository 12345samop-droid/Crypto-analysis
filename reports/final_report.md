# Quantitative Research Report: Next-Day Dominant Side Prediction

## 1. Exact Target Definition
For each asset and for each valid day `t`, predict which extreme of day `t+1` is farther from day `t+1` open.
- Distance Up = `next_day_high - next_day_open`
- Distance Down = `next_day_open - next_day_low`
- **Target = 1** if `Distance Up > Distance Down`
- **Target = 0** if `Distance Down > Distance Up`
- **Ties** were strictly excluded from training and validation sets to prevent bias.

## 2. Daily Alignment Logic
The boundary of a "day" is strictly defined using the date portion of the `open_time` timestamp (UTC+05:30). Aggregations:
- `open` = first 4H open of the date
- `high` = maximum 4H high of the date
- `low` = minimum 4H low of the date
- `close` = last 4H close of the date
- `volume` = sum of all 4H volumes of the date

## 3. Daily Filtering Logic
- Days containing fewer than 6 four-hour candles were discarded.
- Samples where `t+1` is partial or invalid were dropped to ensure accurate labels.

## 4. Feature Engineering Summary
Features are constructed strictly using information up to the end of day `t`:
- **Candle Structure:** Range ratios, body ratios, wick ratios, close location within daily range.
- **Multi-day Context:** Rolling means, volatility indices (rolling range/average range), and distance from rolling highs/lows.
- **Cross-asset (BTC) Context:** BTC's rolling volatility regime and daily return.
- **Calendar effects:** Weekday, month, quarter.

## 5. Validation Design
A chronologically split Walk-Forward Validation (Expanding Window) was implemented to replicate live out-of-sample conditions. Ties were excluded from binary evaluation.
We used 5 splits to robustly evaluate performance decay. Random label permutation tests and asset stability checks were performed on the final holdout.

## 6. Best Model / Results
We tested several baselines (Majority Class, Momentum, Reversion), Logistic Regression, and LightGBM.

**In-sample (LightGBM on first 70%):**
- Accuracy: 0.77
- ROC-AUC: 0.865

**Out-of-sample (Test sets across walk-forward splits):**
- Split 1: AUC = 0.514, Acc = 0.516
- Split 2: AUC = 0.589, Acc = 0.572
- Split 3: AUC = 0.464, Acc = 0.471
- Split 4: AUC = 0.497, Acc = 0.497
- Split 5: AUC = 0.536, Acc = 0.529
- **Mean Walk-Forward AUC (LGBM):** ~0.520
- **Mean Walk-Forward AUC (LogReg):** ~0.523

## 7. Robustness and Adversarial Results
- The Out-of-sample AUC heavily fluctuated across splits (0.46 to 0.58).
- Randomized Labels check in Split 5 yielded an AUC of 0.513 vs true label AUC of 0.536, showing a very marginal edge.
- Asset Stability check on the final split showed the model predicts well on some (SOL AUC=0.60, LTC AUC=0.58) but fails completely on others (TRX AUC=0.44, XRP AUC=0.46, MATIC AUC=0.48).

## 8. Final Judgment: REJECTED
While there are isolated pockets of predictive power (e.g., in highly volatile assets or specific bullish time-windows), there is **no consistently stable, durable, cross-asset edge**. Out-of-sample performance collapses into random noise (~0.50 AUC) for several time slices and completely inverted on some assets. Given the structural testing requirements, the dataset and feature families explored do not support a tradable claim.

**Recommendation:** The hypothesis of predicting the dominant next-day excursion using pure price-action and standard technical regime markers on this 4H crypto dataset is rejected.
