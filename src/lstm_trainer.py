"""
Standalone LSTM trainer for water inflow forecasting.

Runs in a separate process to avoid PyTorch/Jupyter kernel conflicts on macOS.
Called from notebook via subprocess; results saved as pickle.

Usage:
    python src/lstm_trainer.py --data_path data/multistep_regression.csv --output_path /tmp/lstm_results.pkl
"""

import argparse
import gc
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# Use centralized logger
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_sequences(values, seq_len):
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len:i])
        y.append(values[i])
    return np.array(X).reshape(-1, seq_len, 1), np.array(y)


def compute_mase(actual, predicted, train_series, seasonal_period=12):
    naive_errors = np.abs(train_series.values[seasonal_period:] - train_series.values[:-seasonal_period])
    scale = np.mean(naive_errors)
    if scale == 0:
        return np.nan
    return np.mean(np.abs(actual - predicted)) / scale


def compute_metrics(actual, predicted, train_series=None):
    actual = np.array(actual)
    predicted = np.array(predicted)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAE': mean_absolute_error(actual, predicted),
        'MAPE': mean_absolute_percentage_error(actual, predicted) * 100,
    }
    if train_series is not None:
        metrics['MASE'] = compute_mase(actual, predicted, train_series)
    return metrics


def get_cv_folds(series, first_test_year=2018, last_test_year=2022, test_months=5):
    folds = []
    for test_year in range(first_test_year, last_test_year + 1):
        dec_date = pd.Timestamp(f'{test_year - 1}-12-01')
        jan_date = pd.Timestamp(f'{test_year}-01-01')
        may_date = pd.Timestamp(f'{test_year}-{test_months:02d}-01')
        if dec_date not in series.index or may_date not in series.index:
            continue
        train_fold = series[:dec_date]
        test_fold = series[jan_date:may_date]
        folds.append((train_fold, test_fold, f'Dec {test_year - 1} -> Jan-May {test_year}'))
    return folds


def load_series(data_path):
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))
    df = df.sort_values('date').set_index('date')
    return df['Value']


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_one_fold(train_fold_series, test_fold_series, label, fold_idx, n_folds,
               seq_len=12, max_epochs=150, patience=10, hidden_size=32, test_size=5):
    logger.info("Fold %d/%d: %s", fold_idx + 1, n_folds, label)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_fold_series.values.reshape(-1, 1)).flatten()

    X_np, y_np = build_sequences(train_scaled, seq_len)
    X_t = torch.FloatTensor(X_np)
    y_t = torch.FloatTensor(y_np)
    del X_np, y_np

    # Train/val split for early stopping
    val_split = max(len(X_t) - 12, int(len(X_t) * 0.85))
    X_train, X_val = X_t[:val_split], X_t[val_split:]
    y_train, y_val = y_t[:val_split], y_t[val_split:]
    del X_t, y_t

    torch.manual_seed(42)
    model = LSTMNet(input_size=1, hidden_size=hidden_size, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    stopped_epoch = max_epochs

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train).squeeze(), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val).squeeze(), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            stopped_epoch = epoch + 1
            break

    model.load_state_dict(best_state)
    del best_state, X_train, X_val, y_train, y_val, optimizer
    gc.collect()

    # Recursive prediction
    model.eval()
    history = list(train_scaled)
    preds_scaled = []

    with torch.no_grad():
        for _ in range(test_size):
            seq = torch.FloatTensor(np.array(history[-seq_len:]).reshape(1, seq_len, 1))
            pred = float(model(seq).squeeze())
            preds_scaled.append(pred)
            history.append(pred)

    fold_pred = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    fold_metrics = compute_metrics(test_fold_series.values, fold_pred, train_fold_series)

    logger.info("  Early stopped at epoch %d/%d (patience=%d)", stopped_epoch, max_epochs, patience)
    logger.info("  Best val loss: %.6f", best_val_loss)
    for k, v in fold_metrics.items():
        logger.info("  %s: %.3f", k, v)

    del model, criterion
    gc.collect()

    return fold_pred, fold_metrics, stopped_epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='LSTM Walk-Forward Trainer')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--test_size', type=int, default=5)
    parser.add_argument('--first_test_year', type=int, default=2018)
    parser.add_argument('--last_test_year', type=int, default=2022)
    args = parser.parse_args()

    ts = load_series(args.data_path)
    cv_folds = get_cv_folds(ts, args.first_test_year, args.last_test_year, args.test_size)

    logger.info("LSTM Walk-Forward Cross-Validation (%d folds)", len(cv_folds))

    fold_metrics_list = []
    fold_predictions = {}

    for fold_idx, (train_fold, test_fold, label) in enumerate(cv_folds):
        fold_pred, fold_metrics, stopped_epoch = train_one_fold(
            train_fold, test_fold, label, fold_idx, len(cv_folds),
            seq_len=args.seq_len, max_epochs=args.max_epochs,
            patience=args.patience, hidden_size=args.hidden_size,
            test_size=args.test_size,
        )
        fold_metrics_list.append(fold_metrics)
        fold_predictions[fold_idx] = {
            'pred': fold_pred,
            'actual_values': test_fold.values,
            'actual_index': test_fold.index,
            'train_values': train_fold.values,
            'train_index': train_fold.index,
        }

    # CV Summary
    cv_summary = {}
    for metric in ['RMSE', 'MAE', 'MAPE', 'MASE']:
        values = [m[metric] for m in fold_metrics_list]
        cv_summary[metric] = {'mean': np.mean(values), 'std': np.std(values)}

    logger.info("LSTM CV Summary:")
    for metric, stats in cv_summary.items():
        logger.info("  %s: %.3f +/- %.3f", metric, stats['mean'], stats['std'])

    # Save results
    results = {
        'fold_metrics': fold_metrics_list,
        'fold_predictions': fold_predictions,
        'cv_summary': cv_summary,
        'final_pred': fold_predictions[len(cv_folds) - 1]['pred'],
        'final_metrics': fold_metrics_list[-1],
    }

    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info("Results saved to %s", args.output_path)


if __name__ == '__main__':
    main()
