# A Project of Time Series Forecasting in Supply Chain Analysis | Application from the book "Data Science for Supply Chain Forecasting"

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-yellow)

A comprehensive, educational Jupyter notebook exploring classical **time series forecasting techniques**, with a strong focus on **exponential smoothing methods**. This project systematically builds up forecasting knowledge — from simple moving averages to advanced **triple exponential smoothing (Holt-Winters)** with **damped trends**, **multiplicative/additive seasonality**, outlier treatment, and model parameter optimization.

## Project Description | Supply_Chain_Statistical_Forecasting

This repository contains a single, well-commented Jupyter notebook (`Time_Series_Forecasting_Exploration.ipynb`) that demonstrates:

- Synthetic monthly demand data generation (trend + seasonality + noise)
- Common forecasting accuracy KPIs (Bias, MAE, MAPE, MSE, RMSE)
- Implementation and visualization of multiple forecasting methods
- Step-by-step progression from basic to sophisticated techniques
- Parameter optimization for exponential smoothing models
- Outlier detection & treatment using Winsorization and statistical limits
- Comparison of additive vs. multiplicative seasonality in Holt-Winters models

### Forecasting Methods Covered

1. **Naive / Moving Average** (simple baseline)
2. **Simple Exponential Smoothing** (SES) – level only
3. **Double Exponential Smoothing** (Holt’s linear trend) – level + trend
4. **Double Exponential Smoothing with Damped Trend** – prevents explosive long-term forecasts
5. **Triple Exponential Smoothing (Holt-Winters Multiplicative)** with damped trend
6. **Triple Exponential Smoothing (Holt-Winters Additive)** with damped trend
7. **Grid search optimization** of smoothing parameters (α, β, γ)

### Additional Topics

- KPI calculation function (Bias, MAE, MAPE, MSE, RMSE)
- Outlier handling techniques:
  - Percentile-based Winsorization
  - Standard deviation (normal distribution assumption)
  - Model error-based outlier detection & iterative cleaning
- Visual comparison of actual demand vs. forecasts
- Explanation of when to use additive vs. multiplicative seasonality

## Key Features

- Clean, modular function-based implementation of each method
- Consistent interface: most functions accept `demand`, `fcst_period`, smoothing parameters
- Proper handling of in-sample fit + one-step-ahead forecasts
- Visualizations created with Matplotlib for every major method
- Educational comments explaining the logic, formulas, and trade-offs
- Synthetic but realistic demand pattern (linear trend + yearly cycle + noise)

## Technologies Used

- **Python 3.8+**
- **NumPy** – array operations & random data generation
- **Pandas** – data handling & DataFrame results
- **Matplotlib** – visualization of time series and forecasts
- **SciPy** – normal distribution functions for outlier limits

## How to Use

1. Clone the repository

   ```bash
   git clone https://github.com/YOUR-USERNAME/time-series-forecasting-exponential-smoothing.git
   cd time-series-forecasting-exponential-smoothing

2. Install dependencies (recommended: virtual environment)Bash
   ```bash 
   pip install numpy pandas matplotlib scipy jupyter
3. Launch Jupyter NotebookBash
   ```bash
   jupyter notebook
4. Open Time_Series_Forecasting_Exploration.ipynb and run all cells

You can easily modify the following to experiment:
+ np.random.seed() value
+ Smoothing parameters (alpha, beta, gamma, phi)
+ Forecast horizon (fcst_period)
+ Season length (season_length)
+ Outlier detection thresholds and methods

### Learning Outcomes
By studying this notebook, you will understand:

How exponential smoothing family methods work under the hood
Mathematical formulation of level, trend, and seasonal components
Difference between additive and multiplicative seasonality
Why and when to apply damping to trends
How to evaluate and compare forecasting models quantitatively
Practical treatment of outliers in time series data
Trade-offs between model complexity and forecast stability

### Future Improvements (Ideas)

Add automatic parameter optimization using scipy.optimize or grid + cross-validation
Implement forecast confidence intervals
Add real-world dataset examples (retail, energy, web traffic…)
Include ARIMA/SARIMA comparison
Create a Streamlit or Dash dashboard for interactive forecasting
