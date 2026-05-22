import pandas as pd
import numpy as np
import sys
import os
import lightgbm as lgb
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import TimeSeriesValidation, evaluate_model

def run_tree_model():
    print("Loading features...")
    df = pd.read_csv('data/features.csv')
    df = df[df['tie_flag'] == 0].copy()

    val = TimeSeriesValidation(df)
    train_df, test_df = val.train_test_split(train_ratio=0.7)

    # All features
    features = [c for c in df.columns if c.startswith('f_')]

    X_train = train_df[features].values
    y_train = train_df['target'].values

    X_test = test_df[features].values
    y_test = test_df['target'].values

    # LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=features)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=features, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 5,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }

    print("Training LightGBM...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data]
    )

    # Evaluate Train
    y_prob_train = model.predict(X_train)
    y_pred_train = (y_prob_train > 0.5).astype(int)
    print("\n--- In-Sample (Train) ---")
    print(evaluate_model(y_train, y_pred_train, y_prob_train))

    # Evaluate Test
    y_prob_test = model.predict(X_test)
    y_pred_test = (y_prob_test > 0.5).astype(int)
    print("\n--- Out-of-Sample (Test) ---")
    print(evaluate_model(y_test, y_pred_test, y_prob_test))

    # Feature Importance
    importance = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': features, 'gain': importance}).sort_values('gain', ascending=False)
    print("\n--- Top 10 Features ---")
    print(imp_df.head(10))

    os.makedirs('figures', exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.barh(imp_df['feature'].head(20)[::-1], imp_df['gain'].head(20)[::-1])
    plt.title('Top 20 Features by Gain (LightGBM)')
    plt.tight_layout()
    plt.savefig('figures/lgbm_importance.png')

if __name__ == '__main__':
    run_tree_model()
