# Water Inflow Forecasting

Multi-step time series forecasting system for predicting monthly water inflow
at a hydroelectric power plant. The system generates 5-month ahead forecasts
(January–May) every December, combining four complementary models into a
weighted ensemble with uncertainty quantification.

## Models

| Model | Type | Description |
|-------|------|-------------|
| Holt-Winters | Statistical | Triple exponential smoothing with multiplicative seasonality (period=12) |
| XGBoost | Machine Learning | Gradient boosting on lag/rolling features with linear detrending |
| Random Forest | Machine Learning | Bagging ensemble on the same engineered feature set |
| LSTM | Deep Learning | Single-layer LSTM (PyTorch) with MinMax scaling and early stopping |

Ensemble weights are optimized by minimizing MAE on pooled walk-forward
cross-validation residuals. Prediction intervals are computed via **conformal
prediction** at 80%, 90%, and 95% confidence levels, scaled by forecast horizon.

## Project Structure

```
water-inflow-forecasting/
├── data/
│   ├── PROBLEM.md                  # Problem definition and task requirements
│   ├── multistep_regression.csv    # Raw monthly time series (1999-2022, 281 obs)
│   ├── features_all.csv            # All engineered features (281 rows, 30 cols, has NaN)
│   └── features_clean.csv          # Clean features after dropna (257 rows, first 24 dropped due to lag_24)
│
├── src/
│   ├── water_inflow_forecaster.py  # Main forecaster class (fit/predict/evaluate/save/load)
│   ├── lstm_trainer.py             # Standalone LSTM trainer (runs as subprocess)
│   ├── feature_engineering_comparison.py  # FE impact analysis script
│   └── utils/
│       ├── __init__.py
│       └── logger.py               # Centralized logging (console + daily file)
│
├── notebooks/
│   ├── 01_EDA_and_Feature_Engineering.ipynb  # Exploratory analysis & feature creation
│   ├── 02_Model_Development.ipynb            # Model training, CV, ensemble & intervals
│   └── 03_Feature_Engineering_Impact.ipynb   # FE impact: 3 scenarios compared
│
├── figures/                        # All generated visualizations (27 PNG + 1 JSON)
├── models/                         # Serialized model artifacts
│   └── forecaster.pkl              # Trained WaterInflowForecaster instance
├── docs/                           # Reports and presentations
│   ├── report.docx                #   Project report (Word)
│   └── presentation.pptx         #   Project presentation (PowerPoint)
├── logs/                           # Daily execution logs (auto-generated)
│
├── requirements.txt
├── pyproject.toml                  # Ruff linter & formatter configuration
├── .gitignore
└── LICENSE
```

## Data

`data/multistep_regression.csv` contains monthly water inflow records:

| Column | Description |
|--------|-------------|
| `Year` | Observation year (1999–2022) |
| `Month` | Observation month (1–12) |
| `Value` | Monthly water inflow (m³/s) |

The engineered feature files add lag features (1–24 months), rolling statistics
(mean/std/min/max at windows 3, 6, 12), seasonal encoding (sin/cos of month),
differencing (1-month and 12-month), and a normalized time index.

## Setup

### Prerequisites

- Python 3.12+
- pip

### Create virtual environment and install dependencies

```bash
# Clone the repository
git clone https://github.com/<your-username>/water-inflow-forecasting.git
cd water-inflow-forecasting

# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import torch, sklearn, xgboost, statsmodels; print('All dependencies OK')"
```

## Usage

### 1. Explore the notebooks

Launch Jupyter and run the notebooks in order:

```bash
jupyter notebook notebooks/
```

| Notebook | Purpose |
|----------|---------|
| `01_EDA_and_Feature_Engineering` | Time series decomposition, stationarity tests, ACF/PACF analysis, feature engineering pipeline |
| `02_Model_Development` | Train all 4 models, walk-forward CV, ensemble optimization, prediction intervals, residual diagnostics |
| `03_Feature_Engineering_Impact` | Compare 3 scenarios (raw lags vs full FE vs PACF-guided), quantify improvement |

### 2. Use the forecaster programmatically

```python
import pandas as pd
from src.water_inflow_forecaster import WaterInflowForecaster

# Load data
df = pd.read_csv("data/multistep_regression.csv")

# Train
forecaster = WaterInflowForecaster(forecast_horizon=5, confidence_level=0.95)
forecaster.fit(df)

# Predict next 5 months
result = forecaster.predict()
print(result.summary())

# Access predictions and intervals
print(result.predictions)          # Point forecasts
print(result.lower_bound)          # 95% CI lower
print(result.upper_bound)          # 95% CI upper
print(result.confidence_levels)    # 80%, 90%, 95% intervals

# Save / load
forecaster.save("models/forecaster.pkl")
loaded = WaterInflowForecaster.load("models/forecaster.pkl")
```

### 3. Run standalone scripts

```bash
# Feature engineering impact analysis (generates 2 figures + metrics JSON)
python src/feature_engineering_comparison.py

# LSTM trainer as subprocess (used internally by notebooks)
python src/lstm_trainer.py \
    --data_path data/multistep_regression.csv \
    --output_path /tmp/lstm_results.pkl
```

## Source Code Details

### `src/water_inflow_forecaster.py`

The main production module. Key components:

- **`ForecastResult`** — Dataclass holding point forecasts, prediction intervals
  (multiple confidence levels), per-model predictions, and evaluation metrics.
- **`WaterInflowForecaster`** — End-to-end forecaster with:
  - `fit(data, validation_size=5)` — Trains all models using December-anchored
    walk-forward CV, optimizes ensemble weights, calibrates conformal intervals.
  - `predict(steps=None)` — Generates multi-step forecasts with uncertainty bands.
  - `evaluate(actual, predicted)` — Computes RMSE, MAE, MAPE, SMAPE, and
    residual diagnostics (Shapiro-Wilk normality test, lag-1 autocorrelation).
  - `save(path)` / `load(path)` — Model persistence via pickle.

### `src/lstm_trainer.py`

Standalone LSTM training script designed to run as a **subprocess** to avoid
PyTorch/Jupyter kernel conflicts on macOS. Implements its own walk-forward CV
(5 folds, 2018–2022) and outputs fold-level metrics (RMSE, MAE, MAPE, MASE).

### `src/feature_engineering_comparison.py`

Compares model performance with and without feature engineering:
- **Baseline**: Raw lag features only (lag 1–12)
- **Full FE**: Detrending + rolling statistics + seasonal sin/cos encoding

Trains XGBoost and Random Forest on both scenarios and outputs comparison
bar charts and prediction overlays to `figures/`.

### `src/utils/logger.py`

Centralized logging utility. Creates loggers with dual output:
- Console (stdout)
- Daily rotating log file at `logs/YYYY-MM-DD.log`

```python
from utils.logger import get_logger
logger = get_logger(__name__)
```

## Cross-Validation Strategy

The system uses **December-anchored walk-forward cross-validation** that mirrors
the real operational scenario:

```
Fold 1: Train [1999-01 ... 2017-12] → Test [2018-01 ... 2018-05]
Fold 2: Train [1999-01 ... 2018-12] → Test [2019-01 ... 2019-05]
Fold 3: Train [1999-01 ... 2019-12] → Test [2020-01 ... 2020-05]
Fold 4: Train [1999-01 ... 2020-12] → Test [2021-01 ... 2021-05]
Fold 5: Train [1999-01 ... 2021-12] → Test [2022-01 ... 2022-05]
```

## Figures

All visualizations are saved to `figures/`. Key outputs include:

| Figure | Description |
|--------|-------------|
| `c2_timeseries.png` | Full 24-year time series overview |
| `c2_seasonal_decomposition.png` | STL decomposition (trend, seasonal, residual) |
| `c2_acf_pacf.png` | Autocorrelation and partial autocorrelation plots |
| `c2_model_comparison.png` | Performance metrics across all 4 models |
| `c2_residual_diagnostics.png` | Residual distribution and autocorrelation analysis |
| `c2_fe_comparison_bar.png` | Feature engineering impact: metric comparison |
| `c2_fe_comparison_pred.png` | Feature engineering impact: prediction overlay |
| `c2_feature_importance_comparison.png` | Feature importance across XGBoost and RF |

## Documentation

The `docs/` directory contains the project report and presentation:

| File                 | Description                                                           |
|----------------------|-----------------------------------------------------------------------|
| `report.docx`        | Detailed project report covering methodology, results, and analysis   |
| `presentation.pptx`  | Project presentation summarizing key findings and forecasting results |

## License

MIT License
