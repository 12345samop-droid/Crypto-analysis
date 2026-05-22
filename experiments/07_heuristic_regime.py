import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.validation import TimeSeriesValidation

def test_heuristic():
    print("Loading features...")
    df = pd.read_csv('data/features.csv')
    df = df[df['tie_flag'] == 0].copy()

    val = TimeSeriesValidation(df)
    train_df, test_df = val.train_test_split(train_ratio=0.7)

    # Let's test a simple rule: High Volume Expansion + Reversal
    # If today had a huge volume spike (>1.5x average) and closed down, does it expand up tomorrow?

    def apply_rule(data):
        # Condition: High volume spike & Down Day
        cond = (data['f_volume_spike'] > 1.5) & (data['f_direction'] == -1)
        # Condition 2: High volume spike & Up Day
        cond2 = (data['f_volume_spike'] > 1.5) & (data['f_direction'] == 1)

        preds = np.full(len(data), 0.5) # default guess
        preds[cond] = 1.0 # predict distance_up > distance_down
        preds[cond2] = 0.0 # predict distance_down > distance_up

        # for metrics, just threshold at 0.5
        pred_class = (preds > 0.5).astype(int)
        pred_class[preds == 0.5] = 0 # default majority

        return preds, pred_class, cond | cond2

    prob_train, pred_train, mask_train = apply_rule(train_df)
    prob_test, pred_test, mask_test = apply_rule(test_df)

    print("\n--- In-Sample (Train) Heuristic ---")
    print(f"Coverage: {mask_train.sum() / len(train_df):.2%}")
    if mask_train.sum() > 0:
        acc = accuracy_score(train_df['target'][mask_train], pred_train[mask_train])
        print(f"Accuracy on triggered events: {acc:.4f}")

    print("\n--- Out-of-Sample (Test) Heuristic ---")
    print(f"Coverage: {mask_test.sum() / len(test_df):.2%}")
    if mask_test.sum() > 0:
        acc = accuracy_score(test_df['target'][mask_test], pred_test[mask_test])
        print(f"Accuracy on triggered events: {acc:.4f}")

if __name__ == '__main__':
    test_heuristic()
