import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import TimeSeriesValidation, evaluate_model

def run_baselines():
    print("Loading features...")
    df = pd.read_csv('data/features.csv')

    # Exclude ties (tie_flag == 1)
    df = df[df['tie_flag'] == 0].copy()

    val = TimeSeriesValidation(df)
    train_df, test_df = val.train_test_split(train_ratio=0.7)

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Baseline 1: Majority Class
    majority_class = train_df['target'].mode()[0]

    y_true_test = test_df['target'].values
    y_pred_maj = np.full_like(y_true_test, fill_value=majority_class)

    metrics_maj = evaluate_model(y_true_test, y_pred_maj)
    print(f"Majority Class ({majority_class}) Baseline:")
    print(metrics_maj)

    # Baseline 2: Simple Heuristic (Mean Reversion / Momentum)
    # E.g., if today was an up day (direction == 1), predict next day will have larger downward excursion (0)
    # Let's see what works on train set

    y_pred_rev = np.where(test_df['f_direction'] == 1, 0, 1)
    metrics_rev = evaluate_model(y_true_test, y_pred_rev)
    print("Mean Reversion Baseline (predict opposite of today's direction):")
    print(metrics_rev)

    y_pred_mom = np.where(test_df['f_direction'] == 1, 1, 0)
    metrics_mom = evaluate_model(y_true_test, y_pred_mom)
    print("Momentum Baseline (predict same as today's direction):")
    print(metrics_mom)

if __name__ == '__main__':
    run_baselines()
