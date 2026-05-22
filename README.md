# Crypto Next-Day Dominant Side Prediction

This repository contains the quantitative research artifacts for discovering, validating, and testing patterns to predict the next-day dominant excursion (distance up vs distance down) in crypto assets based on 4H candlestick data.

## Project Structure
- `data/`: Contains raw CSV data and generated datasets (CSV files are .gitignored)
- `src/`: Core logic for data preparation and validation
  - `data_preparation.py`: Daily aggregation, intra-day struct extraction, target formulation, and tie handling
  - `feature_engineering.py`: Lookahead-free cross-asset feature creation (including rolling and sequence markers)
  - `validation.py`: Time series splitting and walk-forward validation
- `experiments/`: Experimentation scripts
  - `01_baselines.py`: Majority class and naive heuristic checks
  - `02_logistic_regression.py`: Linear evaluation of price action
  - `03_tree_model.py`: LightGBM model for non-linear interactions
  - `04_robustness.py`: Walk-forward and adversarial robustness testing
  - `05_refined_tree.py`: Incorporating intraday structure and high-confidence filtering
  - `06_deep_dive_sol.py`: Asset-specific model stability testing
  - `07_heuristic_regime.py`: Testing isolated volume-price heuristic rules
- `notebooks/`: Exploration code
- `reports/`: Contains the `final_report.md` detailing exact logic, testing, and final judgment.
- `figures/`: Model importance charts

## Setup and Reproducibility
Install dependencies:
`pip install pandas numpy scikit-learn lightgbm matplotlib`

To run the pipeline from scratch:
```
python src/data_preparation.py
python src/feature_engineering.py
python experiments/01_baselines.py
python experiments/02_logistic_regression.py
python experiments/03_tree_model.py
python experiments/04_robustness.py
python experiments/05_refined_tree.py
python experiments/06_deep_dive_sol.py
python experiments/07_heuristic_regime.py
```

## Results Summary
After extensive walk-forward, asset-stability, and feature iteration testing (ranging from multi-day contexts to detailed intraday sequences), **no stable edge was identified**. The patterns failed to maintain consistent Out-Of-Sample AUC across multiple time splits and assets. See `reports/final_report.md` for full details.
