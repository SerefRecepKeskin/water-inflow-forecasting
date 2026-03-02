"""Feature Engineering Impact Comparison
==========================================
Compares model performance: Raw data (no FE) vs Feature Engineered data.
Produces 2 figures:
  - c2_fe_comparison_bar.png  (bar chart: metric comparison)
  - c2_fe_comparison_pred.png (prediction overlay plot)
"""

import matplotlib

matplotlib.use("Agg")

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
np.random.seed(42)

from utils.logger import get_logger

logger = get_logger(__name__)

# ---- Data Loading ----
df = pd.read_csv("../data/multistep_regression.csv")
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
df = df.set_index("Date").sort_index()
df.index.freq = "MS"
ts = df["Value"].copy()

TEST_SIZE = 12
train_ts = ts[:-TEST_SIZE]
test_ts = ts[-TEST_SIZE:]


# ---- Helper: Metrics ----
def compute_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


# ---- Recursive Predict for tree models ----
def recursive_predict(
    model,
    last_values,
    n_steps,
    feature_names,
    future_dates,
    trend_model=None,
    train_len=None,
):
    """Generic recursive predictor."""
    max_lag = 12
    predictions = []
    history = list(last_values[-max_lag:])

    for step in range(n_steps):
        features = {}
        for lag in range(1, max_lag + 1):
            if lag <= len(history):
                features[f"lag_{lag}"] = history[-lag]
            else:
                features[f"lag_{lag}"] = 0

        h = np.array(history)
        if "rolling_mean_3" in feature_names:
            fallback = np.mean(h)
            features["rolling_mean_3"] = np.mean(h[-3:]) if len(h) >= 3 else fallback
            features["rolling_mean_6"] = np.mean(h[-6:]) if len(h) >= 6 else fallback
            features["rolling_mean_12"] = np.mean(h[-12:]) if len(h) >= 12 else fallback
            features["rolling_std_3"] = np.std(h[-3:], ddof=1) if len(h) >= 3 else 0
            features["rolling_std_6"] = np.std(h[-6:], ddof=1) if len(h) >= 6 else 0

        if "month_sin" in feature_names:
            month = future_dates[step].month
            features["month_sin"] = np.sin(2 * np.pi * month / 12)
            features["month_cos"] = np.cos(2 * np.pi * month / 12)

        if "time_index" in feature_names:
            features["time_index"] = (train_len or len(last_values)) + step

        X_step = pd.DataFrame([{f: features.get(f, 0) for f in feature_names}])[
            feature_names
        ]
        pred = model.predict(X_step)[0]
        predictions.append(pred)
        history.append(pred)

    predictions = np.array(predictions)
    if trend_model is not None and train_len is not None:
        X_future = np.arange(train_len, train_len + n_steps).reshape(-1, 1)
        predictions = predictions + trend_model.predict(X_future)

    return predictions


# ==============================================================
# A) BASELINE: Raw Lag Features Only (no rolling, no sin/cos, no detrending)
# ==============================================================
logger.info("=== Baseline Model (Raw Lags Only) ===")


def create_basic_features(series, max_lag=12):
    df_feat = pd.DataFrame(index=series.index)
    df_feat["target"] = series.values
    for lag in range(1, max_lag + 1):
        df_feat[f"lag_{lag}"] = series.shift(lag)
    return df_feat


feat_basic = create_basic_features(train_ts)
feat_basic = feat_basic.dropna()
basic_cols = [c for c in feat_basic.columns if c != "target"]
X_basic = feat_basic[basic_cols]
y_basic = feat_basic["target"]

# XGBoost - basic
xgb_basic = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    n_jobs=1,
    verbosity=0,
)
xgb_basic.fit(X_basic, y_basic)
basic_pred_xgb = recursive_predict(
    xgb_basic, train_ts.values, TEST_SIZE, basic_cols, test_ts.index
)
basic_xgb_metrics = compute_metrics(test_ts.values, basic_pred_xgb)

# RF - basic
rf_basic = RandomForestRegressor(
    n_estimators=100, max_depth=6, random_state=42, n_jobs=1
)
rf_basic.fit(X_basic, y_basic)
basic_pred_rf = recursive_predict(
    rf_basic, train_ts.values, TEST_SIZE, basic_cols, test_ts.index
)
basic_rf_metrics = compute_metrics(test_ts.values, basic_pred_rf)

logger.info(
    "  XGBoost Basic: RMSE=%.2f, MAE=%.2f, MAPE=%.1f%%",
    basic_xgb_metrics["RMSE"],
    basic_xgb_metrics["MAE"],
    basic_xgb_metrics["MAPE"],
)
logger.info(
    "  RF Basic:      RMSE=%.2f, MAE=%.2f, MAPE=%.1f%%",
    basic_rf_metrics["RMSE"],
    basic_rf_metrics["MAE"],
    basic_rf_metrics["MAPE"],
)


# ==============================================================
# B) FULL FE: Detrending + Rolling + Sin/Cos + Time Index
# ==============================================================
logger.info("=== Full Feature Engineering Model ===")

lr = LinearRegression()
X_t = np.arange(len(train_ts)).reshape(-1, 1)
lr.fit(X_t, train_ts.values)
trend = lr.predict(X_t)
detrended = train_ts.values - trend


def create_full_features(series, detrended_values, max_lag=12):
    df_feat = pd.DataFrame(index=series.index)
    df_feat["target"] = detrended_values
    detr_s = pd.Series(detrended_values, index=series.index)
    for lag in range(1, max_lag + 1):
        df_feat[f"lag_{lag}"] = detr_s.shift(lag)
    for window in [3, 6, 12]:
        df_feat[f"rolling_mean_{window}"] = (
            detr_s.shift(1).rolling(window=window).mean()
        )
    for window in [3, 6]:
        df_feat[f"rolling_std_{window}"] = detr_s.shift(1).rolling(window=window).std()
    month = series.index.month
    df_feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * month / 12)
    df_feat["time_index"] = np.arange(len(series))
    return df_feat


feat_full = create_full_features(train_ts, detrended, max_lag=12)
feat_full = feat_full.dropna()
full_cols = [c for c in feat_full.columns if c != "target"]
X_full = feat_full[full_cols]
y_full = feat_full["target"]

# XGBoost - full FE
xgb_full = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=1,
    verbosity=0,
)
xgb_full.fit(X_full, y_full)
full_pred_xgb = recursive_predict(
    xgb_full,
    detrended,
    TEST_SIZE,
    full_cols,
    test_ts.index,
    trend_model=lr,
    train_len=len(train_ts),
)
full_xgb_metrics = compute_metrics(test_ts.values, full_pred_xgb)

# RF - full FE
rf_full = RandomForestRegressor(
    n_estimators=100, max_depth=6, random_state=42, n_jobs=1
)
rf_full.fit(X_full, y_full)
full_pred_rf = recursive_predict(
    rf_full,
    detrended,
    TEST_SIZE,
    full_cols,
    test_ts.index,
    trend_model=lr,
    train_len=len(train_ts),
)
full_rf_metrics = compute_metrics(test_ts.values, full_pred_rf)

logger.info(
    "  XGBoost Full FE: RMSE=%.2f, MAE=%.2f, MAPE=%.1f%%",
    full_xgb_metrics["RMSE"],
    full_xgb_metrics["MAE"],
    full_xgb_metrics["MAPE"],
)
logger.info(
    "  RF Full FE:      RMSE=%.2f, MAE=%.2f, MAPE=%.1f%%",
    full_rf_metrics["RMSE"],
    full_rf_metrics["MAE"],
    full_rf_metrics["MAPE"],
)


# ==============================================================
# Improvement calculations
# ==============================================================
xgb_rmse_imp = (
    (basic_xgb_metrics["RMSE"] - full_xgb_metrics["RMSE"])
    / basic_xgb_metrics["RMSE"]
    * 100
)
rf_rmse_imp = (
    (basic_rf_metrics["RMSE"] - full_rf_metrics["RMSE"])
    / basic_rf_metrics["RMSE"]
    * 100
)
xgb_mae_imp = (
    (basic_xgb_metrics["MAE"] - full_xgb_metrics["MAE"])
    / basic_xgb_metrics["MAE"]
    * 100
)
rf_mae_imp = (
    (basic_rf_metrics["MAE"] - full_rf_metrics["MAE"]) / basic_rf_metrics["MAE"] * 100
)

logger.info("=== Feature Engineering Impact ===")
logger.info("  XGBoost RMSE improvement: %%%.1f", xgb_rmse_imp)
logger.info("  XGBoost MAE improvement:  %%%.1f", xgb_mae_imp)
logger.info("  RF RMSE improvement:      %%%.1f", rf_rmse_imp)
logger.info("  RF MAE improvement:       %%%.1f", rf_mae_imp)


# ==============================================================
# FIGURE 1: Bar Chart Comparison
# ==============================================================
FIGURES_DIR = "../figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

models = ["XGBoost", "Random Forest"]
x = np.arange(len(models))
width = 0.35

# RMSE
ax = axes[0]
basic_vals = [basic_xgb_metrics["RMSE"], basic_rf_metrics["RMSE"]]
full_vals = [full_xgb_metrics["RMSE"], full_rf_metrics["RMSE"]]
bars1 = ax.bar(
    x - width / 2,
    basic_vals,
    width,
    label="Lag Only (No FE)",
    color="#E74C3C",
    alpha=0.8,
)
bars2 = ax.bar(
    x + width / 2,
    full_vals,
    width,
    label="Full FE (Detrend+Rolling+Sin/Cos)",
    color="#27AE60",
    alpha=0.8,
)
ax.set_ylabel("RMSE", fontsize=12)
ax.set_title("RMSE Comparison", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=9)
for bar, val in zip(bars1, basic_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.3,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
for bar, val in zip(bars2, full_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.3,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# MAE
ax = axes[1]
basic_vals = [basic_xgb_metrics["MAE"], basic_rf_metrics["MAE"]]
full_vals = [full_xgb_metrics["MAE"], full_rf_metrics["MAE"]]
bars1 = ax.bar(
    x - width / 2, basic_vals, width, label="Lag Only", color="#E74C3C", alpha=0.8
)
bars2 = ax.bar(
    x + width / 2, full_vals, width, label="Full FE", color="#27AE60", alpha=0.8
)
ax.set_ylabel("MAE", fontsize=12)
ax.set_title("MAE Comparison", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=9)
for bar, val in zip(bars1, basic_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.3,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
for bar, val in zip(bars2, full_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.3,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# MAPE
ax = axes[2]
basic_vals = [basic_xgb_metrics["MAPE"], basic_rf_metrics["MAPE"]]
full_vals = [full_xgb_metrics["MAPE"], full_rf_metrics["MAPE"]]
bars1 = ax.bar(
    x - width / 2, basic_vals, width, label="Lag Only", color="#E74C3C", alpha=0.8
)
bars2 = ax.bar(
    x + width / 2, full_vals, width, label="Full FE", color="#27AE60", alpha=0.8
)
ax.set_ylabel("MAPE (%)", fontsize=12)
ax.set_title("MAPE Comparison", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=9)
for bar, val in zip(bars1, basic_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
for bar, val in zip(bars2, full_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.suptitle(
    "Feature Engineering Impact: Raw Lags vs Full FE",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR + "c2_fe_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close("all")
logger.info("Saved: %sc2_fe_comparison_bar.png", FIGURES_DIR)


# ==============================================================
# FIGURE 2: Prediction Overlay
# ==============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# XGBoost comparison
ax = axes[0]
ax.plot(test_ts.index, test_ts.values, "k-o", markersize=6, linewidth=2, label="Actual")
ax.plot(
    test_ts.index,
    basic_pred_xgb,
    "#E74C3C",
    linestyle="--",
    marker="s",
    markersize=5,
    linewidth=1.5,
    label=f"XGBoost Basic (RMSE={basic_xgb_metrics['RMSE']:.1f})",
)
ax.plot(
    test_ts.index,
    full_pred_xgb,
    "#27AE60",
    linestyle="-",
    marker="D",
    markersize=5,
    linewidth=2,
    label=f"XGBoost Full FE (RMSE={full_xgb_metrics['RMSE']:.1f})",
)
ax.set_title("XGBoost: Basic vs Full FE", fontsize=13, fontweight="bold")
ax.set_ylabel("Water Inflow")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)

# RF comparison
ax = axes[1]
ax.plot(test_ts.index, test_ts.values, "k-o", markersize=6, linewidth=2, label="Actual")
ax.plot(
    test_ts.index,
    basic_pred_rf,
    "#E74C3C",
    linestyle="--",
    marker="s",
    markersize=5,
    linewidth=1.5,
    label=f"RF Basic (RMSE={basic_rf_metrics['RMSE']:.1f})",
)
ax.plot(
    test_ts.index,
    full_pred_rf,
    "#27AE60",
    linestyle="-",
    marker="D",
    markersize=5,
    linewidth=2,
    label=f"RF Full FE (RMSE={full_rf_metrics['RMSE']:.1f})",
)
ax.set_title("Random Forest: Basic vs Full FE", fontsize=13, fontweight="bold")
ax.set_ylabel("Water Inflow")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)

plt.suptitle(
    "Feature Engineering Impact: Prediction Comparison",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR + "c2_fe_comparison_pred.png", dpi=150, bbox_inches="tight")
plt.close("all")
logger.info("Saved: %sc2_fe_comparison_pred.png", FIGURES_DIR)

# Save metrics for PPT/report use
import json

results = {
    "basic_xgb": basic_xgb_metrics,
    "basic_rf": basic_rf_metrics,
    "full_xgb": full_xgb_metrics,
    "full_rf": full_rf_metrics,
    "xgb_rmse_improvement_pct": xgb_rmse_imp,
    "rf_rmse_improvement_pct": rf_rmse_imp,
    "xgb_mae_improvement_pct": xgb_mae_imp,
    "rf_mae_improvement_pct": rf_mae_imp,
}
with open(FIGURES_DIR + "fe_comparison_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
logger.info("Saved: %sfe_comparison_metrics.json", FIGURES_DIR)

logger.info("Done!")
