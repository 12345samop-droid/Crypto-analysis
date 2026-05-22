import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import TimeSeriesValidation, evaluate_model

def walk_forward_test():
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

        # Logistic Regression
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_s, y_train)
        y_prob_lr = lr.predict_proba(X_test_s)[:, 1]

        # LGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'learning_rate': 0.05, 'max_depth': 3, 'verbose': -1, 'random_state': 42
        }
        lgbm = lgb.train(params, train_data, num_boost_round=50)
        y_prob_lgb = lgbm.predict(X_test)

        metrics_lr = evaluate_model(y_test, (y_prob_lr > 0.5).astype(int), y_prob_lr)
        metrics_lgb = evaluate_model(y_test, (y_prob_lgb > 0.5).astype(int), y_prob_lgb)

        results.append({
            'split': i+1,
            'train_dates': f"{train_df['date'].min()} to {train_df['date'].max()}",
            'test_dates': f"{test_df['date'].min()} to {test_df['date'].max()}",
            'lr_auc': metrics_lr['roc_auc'],
            'lr_acc': metrics_lr['accuracy'],
            'lgb_auc': metrics_lgb['roc_auc'],
            'lgb_acc': metrics_lgb['accuracy']
        })

    print("\n--- Walk Forward Results ---")
    res_df = pd.DataFrame(results)
    print(res_df)

    print("\nMean AUC (LR):", res_df['lr_auc'].mean())
    print("Mean AUC (LGB):", res_df['lgb_auc'].mean())

    # Randomization Test on the last split (LGBM)
    last_train, last_test = splits[-1]
    y_train_rand = np.random.permutation(last_train['target'].values)

    train_data_rand = lgb.Dataset(last_train[features].values, label=y_train_rand)
    lgbm_rand = lgb.train(params, train_data_rand, num_boost_round=50)
    y_prob_rand = lgbm_rand.predict(last_test[features].values)
    metrics_rand = evaluate_model(last_test['target'].values, (y_prob_rand > 0.5).astype(int), y_prob_rand)

    print("\n--- Randomized Label Test (Split 5) ---")
    print(f"LGBM True AUC: {res_df.iloc[-1]['lgb_auc']:.4f}")
    print(f"LGBM Rand AUC: {metrics_rand['roc_auc']:.4f}")

    # Check asset stability
    print("\n--- Asset Stability (Split 5) ---")
    last_test['prob_lgb'] = y_prob_lgb
    for asset, grp in last_test.groupby('asset'):
        auc = roc_auc_score(grp['target'], grp['prob_lgb'])
        acc = accuracy_score(grp['target'], (grp['prob_lgb'] > 0.5).astype(int))
        print(f"{asset}: AUC={auc:.4f}, ACC={acc:.4f}")

if __name__ == '__main__':
    walk_forward_test()
