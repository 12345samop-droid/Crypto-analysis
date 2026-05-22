import pandas as pd
import numpy as np
import sys
import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import TimeSeriesValidation, evaluate_model

def refined_walk_forward():
    print("Loading features...")
    df = pd.read_csv('data/features.csv')
    df = df[df['tie_flag'] == 0].copy()

    val = TimeSeriesValidation(df)
    splits = val.walk_forward_splits(n_splits=5, train_size=0.5)

    features = [c for c in df.columns if c.startswith('f_')]

    results = []

    for i, (train_df, test_df) in enumerate(splits):
        X_train = train_df[features].values
        y_train = train_df['target'].values
        X_test = test_df[features].values
        y_test = test_df['target'].values

        # Train generic LGBM with tighter regularization
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'learning_rate': 0.03, 'max_depth': 4, 'num_leaves': 15,
            'feature_fraction': 0.7, 'min_data_in_leaf': 100,
            'verbose': -1, 'random_state': 42
        }

        lgbm = lgb.train(params, train_data, num_boost_round=60)
        y_prob = lgbm.predict(X_test)

        # High confidence filtering
        high_conf_mask = (y_prob > 0.53) | (y_prob < 0.47)
        if high_conf_mask.sum() > 0:
            y_test_hc = y_test[high_conf_mask]
            y_prob_hc = y_prob[high_conf_mask]
            auc_hc = roc_auc_score(y_test_hc, y_prob_hc)
            acc_hc = accuracy_score(y_test_hc, (y_prob_hc > 0.5).astype(int))
            coverage = high_conf_mask.sum() / len(y_test)
        else:
            auc_hc, acc_hc, coverage = np.nan, np.nan, 0.0

        metrics = evaluate_model(y_test, (y_prob > 0.5).astype(int), y_prob)

        results.append({
            'split': i+1,
            'auc_all': metrics['roc_auc'],
            'acc_all': metrics['accuracy'],
            'auc_hc': auc_hc,
            'acc_hc': acc_hc,
            'coverage_hc': coverage
        })

    print("\n--- Refined Walk Forward Results ---")
    res_df = pd.DataFrame(results)
    print(res_df)

    print("\nMean AUC (All):", res_df['auc_all'].mean())
    print("Mean AUC (High Conf):", res_df['auc_hc'].mean())

    # Asset Specific Check on split 5
    print("\n--- Asset-Specific Modeling on Last Split ---")
    train_df, test_df = splits[-1]

    asset_results = []
    for asset in df['asset'].unique():
        a_train = train_df[train_df['asset'] == asset]
        a_test = test_df[test_df['asset'] == asset]

        if len(a_train) < 100 or len(a_test) < 50:
            continue

        a_train_data = lgb.Dataset(a_train[features].values, label=a_train['target'].values)
        a_model = lgb.train(params, a_train_data, num_boost_round=40)

        a_prob = a_model.predict(a_test[features].values)
        a_auc = roc_auc_score(a_test['target'].values, a_prob)
        a_acc = accuracy_score(a_test['target'].values, (a_prob > 0.5).astype(int))

        asset_results.append({
            'asset': asset,
            'auc': a_auc,
            'acc': a_acc
        })

    print(pd.DataFrame(asset_results))

if __name__ == '__main__':
    refined_walk_forward()
