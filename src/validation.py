import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score

class TimeSeriesValidation:
    """
    Validation framework for Time Series data.
    Ensures strict chronological splits with no lookahead.
    """
    def __init__(self, df, date_col='date'):
        self.df = df.copy()
        self.date_col = date_col
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df.sort_values(by=self.date_col, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Extract unique dates and sort them
        self.dates = np.sort(self.df[self.date_col].unique())

    def train_test_split(self, train_ratio=0.7):
        """
        Split by time: first `train_ratio` proportion of dates into train, rest into test.
        """
        split_idx = int(len(self.dates) * train_ratio)
        split_date = self.dates[split_idx]

        train_df = self.df[self.df[self.date_col] < split_date].copy()
        test_df = self.df[self.df[self.date_col] >= split_date].copy()

        return train_df, test_df

    def walk_forward_splits(self, n_splits=5, train_size=0.5):
        """
        Walk forward validation over time.
        Yields train_df, test_df
        """
        total_dates = len(self.dates)
        train_dates_len = int(total_dates * train_size)
        test_dates_len = (total_dates - train_dates_len) // n_splits

        splits = []
        for i in range(n_splits):
            train_end_idx = train_dates_len + i * test_dates_len
            test_end_idx = train_end_idx + test_dates_len
            if i == n_splits - 1:
                test_end_idx = total_dates # Take the rest

            train_end_date = self.dates[train_end_idx]
            test_end_date = self.dates[min(test_end_idx, total_dates - 1)]

            # Using expanding window
            train_df = self.df[self.df[self.date_col] < train_end_date].copy()
            test_df = self.df[(self.df[self.date_col] >= train_end_date) &
                              (self.df[self.date_col] < test_end_date)].copy()

            splits.append((train_df, test_df))

        return splits

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Standard metrics evaluation.
    Ties should be excluded from these metrics.
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = np.nan
            metrics['brier_score'] = np.nan

    return metrics
