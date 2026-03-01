# Water Inflow Forecasting

A time series forecasting project for predicting water inflow using machine learning models including Holt-Winters, Random Forest, and XGBoost.

## Project Structure

```
├── data/                   # Data files
│   ├── multistep_regression.csv   # Raw time series data
│   ├── features_all.csv           # All engineered features
│   └── features_clean.csv         # Cleaned features
├── notebooks/              # Jupyter notebooks for analysis
│   ├── 01_EDA_and_Feature_Engineering.ipynb
│   ├── 02_Model_Development.ipynb
│   ├── 03_Ensemble_and_Uncertainty.ipynb
│   └── 04_Feature_Engineering_Impact.ipynb
├── src/                    # Source code
│   ├── water_inflow_forecaster.py
│   ├── feature_engineering_comparison.py
│   └── regenerate_lag_figures.py
├── figures/                # Generated plots and visualizations
├── reports/                # Reports, presentations, and documentation
├── requirements.txt        # Python dependencies
└── LICENSE
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Start with the notebooks in `notebooks/` for exploratory analysis
2. Use `src/water_inflow_forecaster.py` for model training and prediction
3. Generated figures are saved to `figures/`

## License

MIT License