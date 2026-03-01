# Hydroelectric Power Plant Water Inflow Forecasting

## Problem Description

A hydroelectric power plant maintains monthly records of water inflow to its dam.
The objective is to forecast water inflow for the **next five months** based on
historical data collected from **1999 to 2022**.

Predictions must be generated every **December** for the subsequent five months
(January through May).

## Dataset

- **File**: `multistep_regression.csv`
- **Columns**: `Year`, `Month`, `Value` (monthly water inflow)
- **Period**: January 1999 - December 2022 (288 observations)
- **Granularity**: Monthly

## Tasks

1. **Regression model**: Develop a regression model in Python to forecast water inflow.

2. **Multiple approaches**: Implement at least two different approaches to the
   forecasting task — a traditional statistical method, a machine learning method,
   and a deep-learning-based method. Compare their performance based on accuracy
   metrics such as RMSE, MAPE, and MAE.

3. **Prediction intervals**: Extend the model to provide prediction intervals or
   confidence intervals for the forecasts. Demonstrate how these intervals help
   in decision-making under uncertainty.

4. **Implementation**: Implement the solution as a Python class containing `fit`
   and `predict` methods, submitted as a `.py` file.

5. **Validation**: Validate the model's performance on the provided dataset.

6. **Documentation**: Use third-party libraries as needed but explain the
   algorithm(s) used comprehensively.
