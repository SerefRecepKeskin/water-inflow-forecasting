"""Water Inflow Forecasting Module for Hydroelectric Power Plants.

This module provides a production-ready forecasting system that combines
Holt-Winters, XGBoost, and Random Forest models with uncertainty quantification
via conformal prediction.

The system is designed for multi-step ahead forecasting of monthly water
inflow values at hydroelectric power plants, supporting configurable
forecast horizons and multiple confidence interval levels.

Key Features:
    - Holt-Winters: Captures level, trend, and seasonal patterns (period=12).
    - XGBoost: Handles nonlinear relationships via lag/rolling features
      with linear detrending for stationarity.
    - Random Forest: Bagging-based ensemble for robust nonlinear regression.
    - LSTM: Deep learning sequence model for capturing complex temporal
      dependencies via PyTorch.
    - Ensemble: Weighted combination of all four models.
    - Uncertainty: Conformal prediction intervals at multiple confidence levels.
    - Residual Diagnostics: Residual statistics and autocorrelation.
    - Persistence: Full model save/load via pickle for deployment.

"""

import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from torch import nn
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Use centralized logger
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Container for forecast results with prediction intervals.

    Attributes
    ----------
    predictions : np.ndarray
        Point forecasts from the weighted model combination.
    lower_bound : np.ndarray
        Lower bound of the default (95 %) prediction interval.
    upper_bound : np.ndarray
        Upper bound of the default (95 %) prediction interval.
    model_details : Dict[str, np.ndarray]
        Individual model predictions keyed by model name.
    confidence_levels : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Prediction intervals at multiple confidence levels (e.g., "80%", "90%", "95%").
        Each value is a tuple of (lower_bound, upper_bound).
    metrics : Optional[Dict[str, float]]
        Evaluation metrics computed on validation data (populated after ``evaluate``).

    """

    predictions: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    model_details: dict[str, np.ndarray]
    confidence_levels: dict[str, tuple[np.ndarray, np.ndarray]]
    metrics: dict[str, float] | None = None

    def summary(self) -> str:
        """Return a human-readable summary of the forecast."""
        lines = [
            "Forecast Summary",
            "=" * 50,
            f"  Horizon           : {len(self.predictions)} steps",
            f"  Mean prediction   : {np.mean(self.predictions):.4f}",
            f"  Prediction range  : [{np.min(self.predictions):.4f}, "
            f"{np.max(self.predictions):.4f}]",
            f"  95% CI width avg  : {np.mean(self.upper_bound - self.lower_bound):.4f}",
            f"  Models used       : {list(self.model_details.keys())}",
        ]
        if self.metrics:
            lines.append("  Validation metrics:")
            for k, v in self.metrics.items():
                lines.append(f"    {k:12s}: {v:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LSTM network
# ---------------------------------------------------------------------------


class _LSTMNet(nn.Module):
    """Single-layer LSTM for univariate time series forecasting."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        output_size: int = 1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ---------------------------------------------------------------------------
# Main forecaster class
# ---------------------------------------------------------------------------


class WaterInflowForecaster:
    """Multi-step water inflow forecasting for hydroelectric plants.

    Combines Holt-Winters, XGBoost, and Random Forest models into a weighted
    ensemble with uncertainty quantification via conformal prediction.

    Parameters
    ----------
    forecast_horizon : int
        Default number of months to forecast ahead.
    models : list of str or None
        Model names to include in the ensemble.  Valid entries are
        ``'hw'``, ``'xgboost'``, ``'rf'``, and ``'lstm'``.
        If ``None``, all four are used.
    confidence_level : float
        Default confidence level for the primary prediction interval.

    Examples
    --------
    >>> forecaster = WaterInflowForecaster(forecast_horizon=5)
    >>> forecaster.fit(training_data)
    >>> result = forecaster.predict(steps=5)
    >>> print(result.predictions)
    >>> print(result.lower_bound, result.upper_bound)

    """

    # Valid model identifiers
    _VALID_MODELS = {"hw", "xgboost", "rf", "lstm"}

    def __init__(
        self,
        forecast_horizon: int = 5,
        models: list[str] | None = None,
        confidence_level: float = 0.95,
    ):
        # Validate inputs
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1.")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1).")

        self.forecast_horizon: int = forecast_horizon
        self.model_names: list[str] = models or ["hw", "xgboost", "rf", "lstm"]
        self.confidence_level: float = confidence_level

        # Validate model names
        unknown = set(self.model_names) - self._VALID_MODELS
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Choose from {self._VALID_MODELS}.")

        # Model storage
        self._hw_model: Any = None
        self._xgboost_model: XGBRegressor | None = None
        self._rf_model: RandomForestRegressor | None = None
        self._lstm_model: nn.Module | None = None
        self._lstm_scaler: MinMaxScaler | None = None
        self._trend_coefs: np.ndarray | None = None
        self._xgb_feature_cols: list[str] | None = None
        self._xgb_best_params: dict[str, Any] | None = None

        # Ensemble weights (optimized during fit)
        self._weights: dict[str, float] = {
            name: 1.0 / len(self.model_names) for name in self.model_names
        }

        # Calibration data for conformal prediction
        self._calibration_residuals: np.ndarray | None = None

        # Training data reference (needed for recursive prediction)
        self._train_data: pd.Series | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.Series | pd.DataFrame,
        validation_size: int = 5,
    ) -> "WaterInflowForecaster":
        """Train all models on historical data with walk-forward CV.

        The training pipeline performs the following steps:

        1. Preprocess input data to a ``pd.Series`` with a datetime index.
        2. Generate December-anchored walk-forward CV folds.
        3. For each fold, train all component models and collect
           validation predictions.
        4. Pool validation residuals across all folds for robust
           ensemble weight optimization and conformal calibration.
        5. Retrain all models on the **full** dataset for deployment.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Historical water inflow data.  If a ``DataFrame``, the method
            looks for an ``'inflow'`` or ``'Value'`` column; otherwise it
            uses the first numeric column.  A datetime index is preferred
            but not required.
        validation_size : int
            Number of months in each test window (default 5 for Jan-May).

        Returns
        -------
        self : WaterInflowForecaster
            The fitted forecaster instance (for method chaining).

        Raises
        ------
        ValueError
            If the data is too short or no valid CV folds can be generated.

        """
        logger.info("Starting model training pipeline...")

        # Step 1: Preprocess
        series = self._preprocess_input(data)
        self._train_data = series.copy()

        # Step 2: Generate December-anchored walk-forward CV folds
        cv_folds = self._generate_december_folds(series, test_months=validation_size)
        logger.info("Generated %d walk-forward CV folds.", len(cv_folds))

        # Step 3: Train each model on each fold, collect predictions
        all_val_actuals: list[np.ndarray] = []
        all_val_preds: dict[str, list[np.ndarray]] = {name: [] for name in self.model_names}

        for fold_idx, (train_fold, val_fold, label) in enumerate(cv_folds):
            logger.info(
                "CV Fold %d/%d: %s (train=%d, val=%d)",
                fold_idx + 1,
                len(cv_folds),
                label,
                len(train_fold),
                len(val_fold),
            )
            fold_steps = len(val_fold)

            if "hw" in self.model_names:
                self._fit_hw(train_fold)
                all_val_preds["hw"].append(self._predict_hw(steps=fold_steps))

            if "xgboost" in self.model_names:
                self._fit_xgboost(train_fold)
                all_val_preds["xgboost"].append(self._predict_xgboost(train_fold, steps=fold_steps))

            if "rf" in self.model_names:
                self._fit_rf(train_fold)
                all_val_preds["rf"].append(self._predict_rf(train_fold, steps=fold_steps))

            if "lstm" in self.model_names:
                self._fit_lstm(train_fold)
                all_val_preds["lstm"].append(self._predict_lstm(train_fold, steps=fold_steps))

            all_val_actuals.append(val_fold.values)

        # Step 4: Pool residuals across all folds for weight optimization
        combined_actual = np.concatenate(all_val_actuals)
        combined_preds: dict[str, np.ndarray] = {
            name: np.concatenate(all_val_preds[name]) for name in self.model_names
        }

        self._optimize_weights(combined_actual, combined_preds)
        logger.info("Optimized ensemble weights: %s", self._weights)

        # Calibration residuals for conformal prediction
        ensemble_combined = self._ensemble_combine(combined_preds)
        self._calibration_residuals = np.abs(combined_actual - ensemble_combined)
        logger.info(
            "Calibration residuals: %d points, mean=%.4f, max=%.4f",
            len(self._calibration_residuals),
            np.mean(self._calibration_residuals),
            np.max(self._calibration_residuals),
        )

        # Step 5: Retrain on full data for production use
        logger.info("Retraining all models on full dataset...")
        if "hw" in self.model_names:
            self._fit_hw(series)
        if "xgboost" in self.model_names:
            self._fit_xgboost(series)
        if "rf" in self.model_names:
            self._fit_rf(series)
        if "lstm" in self.model_names:
            self._fit_lstm(series)

        self._is_fitted = True
        logger.info("Training pipeline complete.")
        return self

    def predict(self, steps: int | None = None) -> ForecastResult:
        """Generate multi-step forecasts with prediction intervals.

        Parameters
        ----------
        steps : int, optional
            Number of steps to forecast.  Defaults to ``forecast_horizon``.

        Returns
        -------
        ForecastResult
            Contains point predictions, prediction intervals at the
            configured confidence level, individual model predictions,
            and intervals at 80 %, 90 %, and 95 % levels.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.

        """
        self._check_is_fitted()

        steps = steps or self.forecast_horizon
        if steps < 1:
            raise ValueError("steps must be >= 1.")

        model_predictions: dict[str, np.ndarray] = {}

        if "hw" in self.model_names:
            model_predictions["hw"] = self._predict_hw(steps=steps)

        if "xgboost" in self.model_names:
            model_predictions["xgboost"] = self._predict_xgboost(self._train_data, steps=steps)

        if "rf" in self.model_names:
            model_predictions["rf"] = self._predict_rf(self._train_data, steps=steps)

        if "lstm" in self.model_names:
            model_predictions["lstm"] = self._predict_lstm(self._train_data, steps=steps)

        # Weighted ensemble
        ensemble_pred = self._ensemble_combine(model_predictions)

        # Prediction intervals using conformal prediction with horizon scaling
        lower, upper = self._conformal_interval(ensemble_pred, steps, self.confidence_level)

        # Multiple confidence levels
        confidence_levels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for level in [0.80, 0.90, 0.95]:
            cl, cu = self._conformal_interval(ensemble_pred, steps, level)
            confidence_levels[f"{int(level * 100)}%"] = (cl, cu)

        return ForecastResult(
            predictions=ensemble_pred,
            lower_bound=lower,
            upper_bound=upper,
            model_details=model_predictions,
            confidence_levels=confidence_levels,
        )

    def evaluate(
        self,
        actual: pd.Series | np.ndarray,
        predicted: np.ndarray | None = None,
        steps: int | None = None,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        If ``predicted`` is not supplied, the method generates a forecast
        of length ``len(actual)`` (or ``steps``) and evaluates against it.

        Parameters
        ----------
        actual : pd.Series or np.ndarray
            Ground truth values.
        predicted : np.ndarray, optional
            Pre-computed predictions.  If ``None``, ``predict`` is called.
        steps : int, optional
            Number of forecast steps (used only when ``predicted`` is ``None``).

        Returns
        -------
        Dict[str, float]
            Dictionary with keys ``'rmse'``, ``'mae'``, ``'mape'``, ``'smape'``.

        """
        actual_arr = np.asarray(actual, dtype=float)
        if predicted is None:
            self._check_is_fitted()
            n = steps or len(actual_arr)
            result = self.predict(steps=n)
            predicted = result.predictions[: len(actual_arr)]

        predicted = np.asarray(predicted, dtype=float)
        min_len = min(len(actual_arr), len(predicted))
        a = actual_arr[:min_len]
        p = predicted[:min_len]

        rmse = float(np.sqrt(mean_squared_error(a, p)))
        mae = float(mean_absolute_error(a, p))

        # MAPE (avoid division by zero)
        nonzero_mask = a != 0
        if nonzero_mask.any():
            mape = float(
                np.mean(np.abs((a[nonzero_mask] - p[nonzero_mask]) / a[nonzero_mask])) * 100
            )
        else:
            mape = float("inf")

        # Symmetric MAPE
        denom = np.abs(a) + np.abs(p)
        nonzero_denom = denom != 0
        if nonzero_denom.any():
            smape = float(
                np.mean(2 * np.abs(a[nonzero_denom] - p[nonzero_denom]) / denom[nonzero_denom])
                * 100
            )
        else:
            smape = float("inf")

        metrics = {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape}

        # Add residual diagnostics
        residual_diag = self.residual_diagnostics(a, p)
        metrics["residual_diagnostics"] = residual_diag

        logger.info(
            "Evaluation metrics: %s",
            {k: v for k, v in metrics.items() if k != "residual_diagnostics"},
        )
        logger.info("Residual diagnostics: %s", residual_diag)
        return metrics

    def residual_diagnostics(
        self,
        actual: pd.Series | np.ndarray,
        predicted: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """Compute residual diagnostic statistics.

        Parameters
        ----------
        actual : array-like
            Ground truth values.
        predicted : array-like
            Predicted values.

        Returns
        -------
        Dict[str, float]
            Dictionary containing residual mean, standard deviation,
            and lag-1 autocorrelation.

        """
        actual_arr = np.asarray(actual, dtype=float)
        predicted_arr = np.asarray(predicted, dtype=float)
        residuals = actual_arr - predicted_arr

        # Residual mean and std
        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        # Lag-1 autocorrelation
        if len(residuals) > 1:
            lag1_corr = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
        else:
            lag1_corr = float("nan")

        return {
            "residual_mean": res_mean,
            "residual_std": res_std,
            "lag1_autocorrelation": lag1_corr,
        }

    def save(self, path: str | Path) -> None:
        """Persist the fitted forecaster to disk.

        All model weights, calibration residuals, and configuration
        are serialized into a single pickle file.

        Parameters
        ----------
        path : str or Path
            Destination file path.  Parent directories are created if
            they do not exist.

        """
        self._check_is_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "weights": self._weights,
            "hw": self._hw_model,
            "xgboost": self._xgboost_model,
            "rf": self._rf_model,
            "lstm": self._lstm_model.state_dict() if self._lstm_model else None,
            "lstm_scaler": self._lstm_scaler,
            "lstm_seq_len": getattr(self, "_lstm_seq_len", 12),
            "xgb_feature_cols": self._xgb_feature_cols,
            "xgb_best_params": self._xgb_best_params,
            "trend_coefs": self._trend_coefs,
            "calibration_residuals": self._calibration_residuals,
            "train_data": self._train_data,
            "config": {
                "forecast_horizon": self.forecast_horizon,
                "models": self.model_names,
                "confidence_level": self.confidence_level,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "WaterInflowForecaster":
        """Load a previously saved forecaster from disk.

        Parameters
        ----------
        path : str or Path
            Path to the pickle file produced by :meth:`save`.

        Returns
        -------
        WaterInflowForecaster
            Fully restored, ready-to-predict forecaster instance.

        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        config = state["config"]
        forecaster = cls(**config)
        forecaster._weights = state["weights"]
        forecaster._hw_model = state.get("hw")
        forecaster._xgboost_model = state["xgboost"]
        forecaster._rf_model = state.get("rf")
        forecaster._lstm_scaler = state.get("lstm_scaler")
        forecaster._lstm_seq_len = state.get("lstm_seq_len", 12)
        if state.get("lstm") is not None:
            lstm_model = _LSTMNet()
            lstm_model.load_state_dict(state["lstm"])
            lstm_model.eval()
            forecaster._lstm_model = lstm_model
        forecaster._xgb_feature_cols = state.get("xgb_feature_cols")
        forecaster._xgb_best_params = state.get("xgb_best_params")
        forecaster._trend_coefs = state["trend_coefs"]
        forecaster._calibration_residuals = state["calibration_residuals"]
        forecaster._train_data = state["train_data"]

        forecaster._is_fitted = True
        logger.info("Model loaded from %s", path)
        return forecaster

    # ------------------------------------------------------------------
    # Private: walk-forward CV
    # ------------------------------------------------------------------

    def _generate_december_folds(
        self,
        series: pd.Series,
        test_months: int = 5,
        min_train_years: int = 15,
    ) -> list[tuple[pd.Series, pd.Series, str]]:
        """Generate December-anchored expanding window CV folds.

        Each fold trains through December of year *Y* and tests on
        January through May of year *Y + 1*.

        Parameters
        ----------
        series : pd.Series with DatetimeIndex
        test_months : int
            Number of months in each test window (default 5 for Jan-May).
        min_train_years : int
            Minimum years of training data required for a fold.

        Returns
        -------
        list of (train_series, test_series, label) tuples

        """
        folds: list[tuple[pd.Series, pd.Series, str]] = []
        years = sorted(set(series.index.year))

        for year in years:
            dec_date = pd.Timestamp(f"{year}-12-01")
            jan_date = pd.Timestamp(f"{year + 1}-01-01")
            may_date = pd.Timestamp(f"{year + 1}-{test_months:02d}-01")

            if dec_date not in series.index or may_date not in series.index:
                continue

            train_fold = series[:dec_date]
            test_fold = series[jan_date:may_date]

            if len(train_fold) < min_train_years * 12:
                continue

            if len(test_fold) != test_months:
                continue

            label = f"Dec {year} -> Jan-May {year + 1}"
            folds.append((train_fold, test_fold, label))

        if not folds:
            raise ValueError("No valid December-anchored CV folds found in the data.")

        logger.info("Generated %d December-anchored CV folds.", len(folds))
        return folds

    # ------------------------------------------------------------------
    # Private: preprocessing
    # ------------------------------------------------------------------

    def _preprocess_input(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        """Convert input to a float ``pd.Series`` with a sorted datetime index.

        The method supports both ``pd.DataFrame`` (with ``'inflow'``,
        ``'Value'``, or a first-column fallback) and raw ``pd.Series``.
        If the data has ``'Year'`` and ``'Month'`` columns but no datetime
        index, a ``DatetimeIndex`` is synthesized.

        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            Raw input data.

        Returns
        -------
        pd.Series
            Cleaned, sorted, float-valued time series.

        """
        if isinstance(data, pd.DataFrame):
            # Build datetime index from Year/Month if present
            if "Year" in data.columns and "Month" in data.columns:
                if not isinstance(data.index, pd.DatetimeIndex):
                    data = data.copy()
                    data["_date"] = pd.to_datetime(data[["Year", "Month"]].assign(Day=1))
                    data = data.set_index("_date").sort_index()

            # Extract the target column
            if "inflow" in data.columns:
                series = data["inflow"]
            elif "Value" in data.columns:
                series = data["Value"]
            else:
                # Fallback to first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("DataFrame contains no numeric columns.")
                series = data[numeric_cols[0]]
                logger.warning(
                    "No 'inflow' or 'Value' column found; using '%s'.",
                    numeric_cols[0],
                )
        else:
            series = data.copy()

        series = series.astype(float)

        # Handle missing values via linear interpolation
        if series.isna().any():
            n_missing = int(series.isna().sum())
            logger.warning("Found %d missing values; interpolating linearly.", n_missing)
            series = series.interpolate(method="linear").bfill().ffill()

        return series

    # ------------------------------------------------------------------
    # Private: Holt-Winters
    # ------------------------------------------------------------------

    def _fit_hw(self, series: pd.Series) -> None:
        """Fit a Holt-Winters (Triple Exponential Smoothing) model.

        Tests additive and multiplicative configurations for both trend
        and seasonality, selecting the best combination by AIC.
        """
        best_aic = np.inf
        best_model = None
        for trend in ("add", "mul"):
            for seasonal in ("add", "mul"):
                try:
                    model = ExponentialSmoothing(
                        series,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=12,
                    ).fit(optimized=True)
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_model = model
                except Exception:
                    continue
        self._hw_model = best_model
        params = best_model.params
        logger.info(
            "Holt-Winters fitted: alpha=%.4f, beta=%.4f, gamma=%.4f, AIC=%.2f",
            params["smoothing_level"],
            params["smoothing_trend"],
            params["smoothing_seasonal"],
            best_aic,
        )

    def _predict_hw(self, steps: int) -> np.ndarray:
        """Generate ``steps`` ahead forecasts from the fitted Holt-Winters model."""
        predictions = self._hw_model.forecast(steps=steps)
        return np.asarray(predictions, dtype=float)

    # ------------------------------------------------------------------
    # Private: XGBoost
    # ------------------------------------------------------------------

    def _create_lag_features(self, series: pd.Series, max_lag: int = 12) -> pd.DataFrame:
        """Create lag, rolling-window, and seasonal features.

        Parameters
        ----------
        series : pd.Series
            Input time series (already detrended for XGBoost usage).
        max_lag : int
            Maximum lag order (default 12).

        Returns
        -------
        pd.DataFrame
            Feature matrix including the target column ``'value'``.

        """
        df = pd.DataFrame({"value": series.values})

        # Lag features (1..max_lag)
        for lag in range(1, max_lag + 1):
            df[f"lag_{lag}"] = df["value"].shift(lag)

        # Rolling statistics (shifted by 1 to avoid leakage)
        shifted = df["value"].shift(1)
        for window in [3, 6, 12]:
            df[f"rolling_mean_{window}"] = shifted.rolling(window=window).mean()
        for window in [3, 6]:
            df[f"rolling_std_{window}"] = shifted.rolling(window=window).std()

        # Seasonal encoding
        if hasattr(series, "index") and isinstance(series.index, pd.DatetimeIndex):
            months = series.index.month.values
        else:
            months = np.arange(len(series)) % 12 + 1

        df["month_sin"] = np.sin(2 * np.pi * months / 12)
        df["month_cos"] = np.cos(2 * np.pi * months / 12)

        # Trend proxy
        df["time_index"] = np.arange(len(df))

        return df

    def _fit_xgboost(self, series: pd.Series, **kwargs) -> None:
        """Fit XGBoost with linear detrending.

        A first-order polynomial is subtracted from the raw series to
        improve stationarity before feature engineering.
        """
        # Linear detrending
        t = np.arange(len(series), dtype=float)
        self._trend_coefs = np.polyfit(t, series.values, deg=1)
        trend = np.polyval(self._trend_coefs, t)
        detrended = series.values - trend

        detrended_series = pd.Series(detrended, index=series.index)
        df = self._create_lag_features(detrended_series)
        df = df.dropna()

        feature_cols = [c for c in df.columns if c != "value"]
        self._xgb_feature_cols = feature_cols

        X = df[feature_cols].values
        y = df["value"].values

        self._xgboost_model = XGBRegressor(
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
        self._xgboost_model.fit(X, y)
        logger.info(
            "XGBoost fitted with %d features on %d samples.",
            len(feature_cols),
            len(y),
        )

    def _predict_xgboost(self, series: pd.Series, steps: int) -> np.ndarray:
        """Generate recursive multi-step XGBoost predictions.

        Each step is predicted one at a time and appended to the history
        so that subsequent steps can use the predicted values as lag inputs.
        """
        t = np.arange(len(series), dtype=float)
        trend = np.polyval(self._trend_coefs, t)
        detrended = series.values - trend

        # Reconstruct month information for seasonal encoding
        if isinstance(series.index, pd.DatetimeIndex):
            months_history = list(series.index.month.values)
        else:
            months_history = list(np.arange(len(series)) % 12 + 1)

        history = list(detrended)
        predictions: list[float] = []

        for step in range(steps):
            future_t = float(len(series) + step)
            future_trend = np.polyval(self._trend_coefs, future_t)

            # Build a temporary series for feature creation
            temp_series = pd.Series(history)
            df = self._create_lag_features(temp_series)

            # Override month encoding for the last row with correct month
            if isinstance(series.index, pd.DatetimeIndex):
                last_date = series.index[-1]
                future_month = ((last_date.month + step) % 12) or 12  # 1-12 range
            else:
                future_month = (months_history[-1] + step) % 12 or 12
            df.iloc[-1, df.columns.get_loc("month_sin")] = np.sin(2 * np.pi * future_month / 12)
            df.iloc[-1, df.columns.get_loc("month_cos")] = np.cos(2 * np.pi * future_month / 12)
            df.iloc[-1, df.columns.get_loc("time_index")] = len(history) - 1

            last_row = df.iloc[[-1]]
            feature_cols = [c for c in last_row.columns if c != "value"]
            X = last_row[feature_cols].values

            pred_detrended = float(self._xgboost_model.predict(X)[0])
            pred = pred_detrended + future_trend
            predictions.append(pred)
            history.append(pred_detrended)

        return np.array(predictions, dtype=float)

    # ------------------------------------------------------------------
    # Private: Random Forest
    # ------------------------------------------------------------------

    def _fit_rf(self, series: pd.Series) -> None:
        """Fit Random Forest with the same detrending as XGBoost.

        Uses the shared ``_create_lag_features`` feature engineering and
        the linear trend coefficients computed during XGBoost fitting.
        """
        # Use existing trend coefficients or compute new ones
        t = np.arange(len(series), dtype=float)
        if self._trend_coefs is None:
            self._trend_coefs = np.polyfit(t, series.values, deg=1)
        trend = np.polyval(self._trend_coefs, t)
        detrended = series.values - trend

        detrended_series = pd.Series(detrended, index=series.index)
        df = self._create_lag_features(detrended_series)
        df = df.dropna()

        feature_cols = [c for c in df.columns if c != "value"]
        # Reuse XGBoost feature cols if available
        if self._xgb_feature_cols is None:
            self._xgb_feature_cols = feature_cols

        X = df[feature_cols].values
        y = df["value"].values

        self._rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=1,
        )
        self._rf_model.fit(X, y)
        logger.info(
            "Random Forest fitted with %d features on %d samples.",
            len(feature_cols),
            len(y),
        )

    def _predict_rf(self, series: pd.Series, steps: int) -> np.ndarray:
        """Generate recursive multi-step Random Forest predictions.

        Uses the same recursive strategy as XGBoost: each step is
        predicted one at a time and appended to the history.
        """
        t = np.arange(len(series), dtype=float)
        trend = np.polyval(self._trend_coefs, t)
        detrended = series.values - trend

        # Reconstruct month information for seasonal encoding
        if isinstance(series.index, pd.DatetimeIndex):
            months_history = list(series.index.month.values)
        else:
            months_history = list(np.arange(len(series)) % 12 + 1)

        history = list(detrended)
        predictions: list[float] = []

        for step in range(steps):
            future_t = float(len(series) + step)
            future_trend = np.polyval(self._trend_coefs, future_t)

            # Build a temporary series for feature creation
            temp_series = pd.Series(history)
            df = self._create_lag_features(temp_series)

            # Override month encoding for the last row with correct month
            if isinstance(series.index, pd.DatetimeIndex):
                last_date = series.index[-1]
                future_month = ((last_date.month + step) % 12) or 12
            else:
                future_month = (months_history[-1] + step) % 12 or 12
            df.iloc[-1, df.columns.get_loc("month_sin")] = np.sin(2 * np.pi * future_month / 12)
            df.iloc[-1, df.columns.get_loc("month_cos")] = np.cos(2 * np.pi * future_month / 12)
            df.iloc[-1, df.columns.get_loc("time_index")] = len(history) - 1

            last_row = df.iloc[[-1]]
            feature_cols = [c for c in last_row.columns if c != "value"]
            X = last_row[feature_cols].values

            pred_detrended = float(self._rf_model.predict(X)[0])
            pred = pred_detrended + future_trend
            predictions.append(pred)
            history.append(pred_detrended)

        return np.array(predictions, dtype=float)

    # ------------------------------------------------------------------
    # Private: LSTM
    # ------------------------------------------------------------------

    def _build_lstm_sequences(
        self, values: np.ndarray, seq_len: int = 12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create input/target sequences for LSTM training.

        Parameters
        ----------
        values : np.ndarray
            Scaled 1-D time series values.
        seq_len : int
            Lookback window length.

        Returns
        -------
        X : np.ndarray of shape (n_samples, seq_len, 1)
        y : np.ndarray of shape (n_samples,)

        """
        X, y = [], []
        for i in range(seq_len, len(values)):
            X.append(values[i - seq_len : i])
            y.append(values[i])
        X = np.array(X).reshape(-1, seq_len, 1)
        y = np.array(y)
        return X, y

    def _fit_lstm(
        self,
        series: pd.Series,
        seq_len: int = 12,
        hidden_size: int = 32,
        num_layers: int = 1,
        max_epochs: int = 150,
        patience: int = 10,
        lr: float = 1e-3,
    ) -> None:
        """Fit an LSTM model on the time series with early stopping.

        The series is first MinMax-scaled to [0, 1], then sliced into
        overlapping windows of length ``seq_len`` for supervised training.
        Early stopping monitors validation loss (last 12 sequences) to
        prevent overfitting on the small dataset.
        """
        import gc

        # Scale
        self._lstm_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self._lstm_scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

        X, y = self._build_lstm_sequences(scaled, seq_len)
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)

        # Train/val split for early stopping
        val_split = max(len(X_t) - 12, int(len(X_t) * 0.85))
        X_train, X_val = X_t[:val_split], X_t[val_split:]
        y_train, y_val = y_t[:val_split], y_t[val_split:]

        # Build model
        torch.manual_seed(42)
        model = _LSTMNet(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
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
        model.eval()
        self._lstm_model = model
        self._lstm_seq_len = seq_len

        del best_state, X_train, X_val, y_train, y_val, X_t, y_t, optimizer
        gc.collect()

        logger.info(
            "LSTM fitted (early stopped at %d/%d epochs, best val loss=%.6f).",
            stopped_epoch,
            max_epochs,
            best_val_loss,
        )

    def _predict_lstm(self, series: pd.Series, steps: int) -> np.ndarray:
        """Generate recursive multi-step LSTM predictions."""
        scaled = self._lstm_scaler.transform(series.values.reshape(-1, 1)).flatten()

        history = list(scaled)
        predictions: list[float] = []

        self._lstm_model.eval()
        with torch.no_grad():
            for _ in range(steps):
                seq = np.array(history[-self._lstm_seq_len :]).reshape(1, self._lstm_seq_len, 1)
                x_t = torch.FloatTensor(seq)
                pred_scaled = float(self._lstm_model(x_t).squeeze())
                history.append(pred_scaled)
                predictions.append(pred_scaled)

        # Inverse-transform
        predictions = self._lstm_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        return predictions

    # ------------------------------------------------------------------
    # Private: ensemble utilities
    # ------------------------------------------------------------------

    def _ensemble_combine(self, predictions_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Combine individual model predictions using learned weights."""
        return sum(self._weights[m] * predictions_dict[m] for m in predictions_dict)

    def _optimize_weights(
        self,
        actual: np.ndarray,
        predictions_dict: dict[str, np.ndarray],
    ) -> None:
        """Optimize ensemble weights by minimizing MSE on the validation set.

        The optimization is constrained so that all weights are in [0.01, 0.99]
        and sum to 1.  This ensures every model contributes at least
        minimally to the ensemble, which improves robustness.

        Parameters
        ----------
        actual : np.ndarray
            Actual validation values.
        predictions_dict : Dict[str, np.ndarray]
            Validation predictions keyed by model name.

        """
        model_names = list(predictions_dict.keys())
        n = len(model_names)

        if n == 0:
            return
        if n == 1:
            self._weights = {model_names[0]: 1.0}
            return

        pred_matrix = np.array([predictions_dict[m] for m in model_names])

        def objective(weights: np.ndarray) -> float:
            ensemble = np.average(pred_matrix, axis=0, weights=weights)
            return float(mean_squared_error(actual, ensemble))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.01, 0.99)] * n
        x0 = np.ones(n) / n

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )

        if result.success:
            for i, name in enumerate(model_names):
                self._weights[name] = float(result.x[i])
        else:
            logger.warning(
                "Weight optimization did not converge (%s). Using equal weights.",
                result.message,
            )
            for name in model_names:
                self._weights[name] = 1.0 / n

    def _conformal_interval(
        self,
        ensemble_pred: np.ndarray,
        steps: int,
        confidence_level: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals using conformal prediction.

        The calibration residuals obtained during ``fit()`` are used to
        determine the quantile width of the interval.  A square-root
        horizon factor widens the interval for further-ahead forecasts,
        reflecting increasing uncertainty.

        Parameters
        ----------
        ensemble_pred : np.ndarray
            Point predictions from the ensemble.
        steps : int
            Forecast horizon (used for horizon-based interval widening).
        confidence_level : float
            Desired coverage probability in (0, 1).

        Returns
        -------
        lower : np.ndarray
            Lower bounds (clipped at 0 since inflow is non-negative).
        upper : np.ndarray
            Upper bounds.

        """
        if self._calibration_residuals is not None and len(self._calibration_residuals) > 0:
            q = float(np.quantile(self._calibration_residuals, confidence_level))
            # Widen intervals for further horizons
            horizon_factor = np.sqrt(np.arange(1, steps + 1, dtype=float))
            lower = ensemble_pred - q * horizon_factor
            upper = ensemble_pred + q * horizon_factor
        else:
            # Fallback: use a percentage-based heuristic
            logger.warning("No calibration residuals available; using heuristic intervals.")
            half_width = (1 - confidence_level) * 0.5
            lower = ensemble_pred * (1 - half_width * 4)
            upper = ensemble_pred * (1 + half_width * 4)

        # Water inflow is physically non-negative
        lower = np.maximum(lower, 0.0)

        return lower, upper

    # ------------------------------------------------------------------
    # Private: validation helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise ``RuntimeError`` if the model has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "This WaterInflowForecaster instance is not fitted yet. "
                "Call 'fit()' with training data before using 'predict()'."
            )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"WaterInflowForecaster("
            f"horizon={self.forecast_horizon}, "
            f"models={self.model_names}, "
            f"confidence={self.confidence_level}, "
            f"status={status})"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "multistep_regression.csv",
    )

    if not os.path.exists(data_path):
        logger.error("Data file not found at %s", data_path)
        raise SystemExit(1)

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    # Build datetime index
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
    df = df.set_index("Date").sort_index()
    df.rename(columns={"Value": "inflow"}, inplace=True)

    logger.info("Data shape: %s, Date range: %s to %s", df.shape, df.index[0], df.index[-1])

    # Initialize and fit
    forecaster = WaterInflowForecaster(forecast_horizon=5)
    logger.info("Forecaster: %s", forecaster)
    forecaster.fit(df, validation_size=5)

    # Predict
    result = forecaster.predict(steps=5)

    logger.info("=" * 60)
    logger.info("5-Month Water Inflow Forecast")
    logger.info("=" * 60)
    logger.info("Predictions : %s", result.predictions)
    logger.info("95%% Lower   : %s", result.lower_bound)
    logger.info("95%% Upper   : %s", result.upper_bound)

    # Individual model contributions
    logger.info("Individual model predictions:")
    for model_name, preds in result.model_details.items():
        weight = forecaster._weights.get(model_name, 0)
        logger.info("  %s (w=%.3f): %s", model_name, weight, preds)

    # Confidence levels
    logger.info("Prediction intervals at multiple levels:")
    for level_str, (lo, hi) in result.confidence_levels.items():
        logger.info("  %s: [%s , %s]", level_str, lo, hi)

    # Summary
    logger.info("\n%s", result.summary())

    # Save and reload test
    save_path = os.path.join(os.path.dirname(__file__), "..", "models", "forecaster.pkl")
    forecaster.save(save_path)

    loaded = WaterInflowForecaster.load(save_path)
    result2 = loaded.predict(steps=5)
    logger.info("Reloaded model predictions: %s", result2.predictions)
    logger.info("Prediction match: %s", np.allclose(result.predictions, result2.predictions, atol=1e-4))
    logger.info("Ensemble Weights: %s", forecaster._weights)
