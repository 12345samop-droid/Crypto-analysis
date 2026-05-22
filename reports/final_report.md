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

## 4. Feature Engineering Summary (Iterated)
Features are constructed strictly using information up to the end of day `t`. Extensive iterations were performed to uncover any structural edge:
- **Candle Structure:** Range ratios, body ratios, wick ratios, close location within daily range.
- **Multi-day Context:** Rolling means, volatility indices (rolling range/average range), and distance from rolling highs/lows.
- **Cross-asset (BTC) Context:** BTC's rolling volatility regime and daily return.
- **Calendar effects:** Weekday, month, quarter.
- **Intraday Sequence (New):** First-half return vs second-half return, indices of the high/low (did high happen before low?), and the return of the final 4H candle of the day.
- **Volume Dynamics (New):** Daily volume spikes relative to recent averages.

## 5. Validation Design
A chronologically split Walk-Forward Validation (Expanding Window) was implemented to replicate live out-of-sample conditions. Ties were excluded from binary evaluation.
We used 5 splits to robustly evaluate performance decay. Asset-specific modeling and high-confidence filtering (thresholding prob > 0.53 or < 0.47) were also utilized to check for conditional edges.

## 6. Best Model / Results
We tested baselines, simple regime heuristics (Volume spikes + Mean reversion), Logistic Regression, and LightGBM models (both pooled and asset-specific).

**In-sample (LightGBM on first 70%):**
- Accuracy: ~0.77
- ROC-AUC: ~0.865

**Out-of-sample (Pooled Models, Refined Iteration):**
- Split 1: AUC = 0.552
- Split 2: AUC = 0.606
- Split 3: AUC = 0.475
- Split 4: AUC = 0.482
- Split 5: AUC = 0.542
- **Mean Walk-Forward AUC:** 0.531

**High-Confidence Filter (Prob > 0.53 or < 0.47):**
- The mean AUC on filtered subsets dropped slightly to 0.527, proving that the model's confidence is severely miscalibrated OOS and does not separate classes better than the unfiltered set.

**Heuristic Baseline (Volume Spike Reversals):**
- Testing rules like "Buy the close on a heavy volume down day" generated a 51.27% out-of-sample accuracy covering ~62% of days. Not significantly better than chance.

## 7. Robustness and Adversarial Results
- The Out-of-sample AUC heavily fluctuated across splits and time periods.
- Randomized Labels check in Split 5 yielded an AUC of 0.51 vs true label AUC of 0.54, showing an extremely thin edge that is likely an artifact of non-stationary volatility regimes.
- Asset Specific Deep Dive (SOL): Modeling SOL purely on its own history peaked at 0.59 AUC in some splits but dropped to 0.46 in others. The overall mean AUC was 0.539, which again is not tradable or consistent.

## 8. Final Judgment: REJECTED
Extensive iteration across multi-day contexts, strict intraday sequences, and volume heuristics confirms that there is **no consistently stable, durable, cross-asset edge** in predicting the dominant next-day excursion from standard 4H price/volume data. Out-of-sample performance frequently collapses into random noise or inverts. The minor localized bumps in AUC (~0.55) fail walk-forward robustness and are non-generalizable across assets.

**Recommendation:** The hypothesis is thoroughly rejected. Any appearance of an edge in-sample is entirely the result of overfitting to specific historical regime correlations.
