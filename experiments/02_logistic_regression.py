import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import TimeSeriesValidation, evaluate_model

def run_logistic_regression():
    print("Loading features...")
    df = pd.read_csv('data/features.csv')
    df = df[df['tie_flag'] == 0].copy()

    val = TimeSeriesValidation(df)
    train_df, test_df = val.train_test_split(train_ratio=0.7)

    # Use only single candle structure features
    features = [
        'f_body_size', 'f_body_range_ratio', 'f_upper_wick_ratio', 'f_lower_wick_ratio',
        'f_close_location', 'f_direction', 'f_range_expansion'
    ]

    X_train = train_df[features].values
    y_train = train_df['target'].values

    X_test = test_df[features].values
    y_test = test_df['target'].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # In-sample
    y_pred_train = model.predict(X_train_scaled)
    y_prob_train = model.predict_proba(X_train_scaled)[:, 1]

    print("--- In-Sample (Train) ---")
    print(evaluate_model(y_train, y_pred_train, y_prob_train))

    # Out-of-sample
    y_pred_test = model.predict(X_test_scaled)
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Out-of-Sample (Test) ---")
    print(evaluate_model(y_test, y_pred_test, y_prob_test))

    print("\n--- Coefficients ---")
    for f, c in zip(features, model.coef_[0]):
        print(f"{f}: {c:.4f}")

if __name__ == '__main__':
    run_logistic_regression()
