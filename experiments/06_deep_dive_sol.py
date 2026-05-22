import pandas as pd
import numpy as np
import sys
import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import TimeSeriesValidation

def test_solana():
    print("Loading features...")
    df = pd.read_csv('data/features.csv')
    df = df[df['tie_flag'] == 0].copy()

    # Isolate SOL
    sol_df = df[df['asset'] == 'SOL'].copy()
    val = TimeSeriesValidation(sol_df)
    splits = val.walk_forward_splits(n_splits=5, train_size=0.5)

    features = [c for c in df.columns if c.startswith('f_')]

    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.02, 'max_depth': 3, 'num_leaves': 7,
        'feature_fraction': 0.8, 'min_data_in_leaf': 30,
        'verbose': -1, 'random_state': 42
    }

    results = []

    for i, (train_df, test_df) in enumerate(splits):
        X_train, y_train = train_df[features].values, train_df['target'].values
        X_test, y_test = test_df[features].values, test_df['target'].values

        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data, num_boost_round=100)

        y_prob = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, (y_prob > 0.5).astype(int))

        results.append({
            'split': i+1,
            'train_dates': f"{train_df['date'].min()} to {train_df['date'].max()}",
            'test_dates': f"{test_df['date'].min()} to {test_df['date'].max()}",
            'auc': auc,
            'acc': acc
        })

        if i == 4: # save importance of last split
            imp = pd.DataFrame({'feat': features, 'gain': model.feature_importance('gain')})
            imp = imp.sort_values('gain', ascending=False)
            print("\nTop 10 features for SOL (Split 5):")
            print(imp.head(10))

    print("\n--- SOL Asset-Specific Walk Forward ---")
    res_df = pd.DataFrame(results)
    print(res_df)

    print(f"\nSOL Mean AUC: {res_df['auc'].mean():.4f}")

    # Let's do a randomized label test on SOL
    train_df, test_df = splits[-1]
    X_train, y_train = train_df[features].values, train_df['target'].values
    y_train_rand = np.random.permutation(y_train)

    train_data_rand = lgb.Dataset(X_train, label=y_train_rand)
    model_rand = lgb.train(params, train_data_rand, num_boost_round=100)

    y_prob_rand = model_rand.predict(test_df[features].values)
    auc_rand = roc_auc_score(test_df['target'].values, y_prob_rand)
    print(f"SOL Randomized Labels AUC (Split 5): {auc_rand:.4f} (True was {res_df.iloc[-1]['auc']:.4f})")

if __name__ == '__main__':
    test_solana()
