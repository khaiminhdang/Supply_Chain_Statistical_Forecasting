# %% 
# # PART 1: STATISTICAL FORECAST

# %% 
# ## 1.1 Data Preprocessing

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create demand array
np.random.seed(42)
months = pd.date_range(start="2023-01-01", periods=24, freq="M")
trend = np.linspace(50, 80, 24)
seasonal = 10 * np.sin(np.linspace(0, 3*np.pi, 24))
noise = np.random.normal(0, 5, 24)
demand = trend + seasonal + noise
demand = demand.round(0)
df = pd.DataFrame({
    "Month": months,
    "Demand": demand})
df

# %% 
# ## 1.2 Forecasting KPIs
def KPI(df):
    error = df["Error"]
    demand = df["Demand"]
    avg_demand = demand.mean()
    mse = (error**2).mean()
    rmse = mse**0.5
    print("Bias:", round(error.mean(), 2))
    print("MAE:", round(error.abs().mean(), 2))
    print("MAPE:", round((error.abs()/demand).mean()*100, 2), "%")
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))

# %%
# ## 1.3 Moving Average Function
def moving_average(demand, fcst_period=1, avg_period=3):    
    forecast = np.full(len(demand), np.nan)
    # Extend demand bằng nan để forecast dc period t+1
    demand = np.append(demand, fcst_period * np.nan)
    forecast = np.append(forecast, fcst_period * np.nan)
    # Tính moving average
    for i in range(avg_period, len(demand)):
        forecast[i] = np.mean(demand[i - avg_period:i])
    df = pd.DataFrame({
        'Demand': demand,
        'Forecast': forecast,
        'Error': demand - forecast})
    return df
df_ma = moving_average(demand, fcst_period=1, avg_period=3)
print(df_ma.head(25))

# KPIs of the forecasting technique
df = moving_average(demand, fcst_period=1, avg_period=3)
KPI(df)

# Visualize the forecasting result
df.index.name = 'Month'
plt.plot(df['Demand'], label='Demand')
plt.plot(df['Forecast'], label='Forecast')
plt.legend()
plt.title('Demand vs Moving Average Forecast')
plt.xlabel('Month')
plt.ylabel('Value')

# %%
# ## 1.4. Simple Exponential Smoothing
def simple_exponential_smoothing(demand, fcst_period=1, alpha=0.4):
    # Create forecast array
    forecast = np.full(len(demand), np.nan)

    # Update demand and forecast array
    demand = np.append(demand, fcst_period * np.nan)
    forecast = np.append(forecast, fcst_period * np.nan)
   
    # Initiate the first forecast
    forecast[0] = demand[0]

    # Forecast the rest
    for i in range(1, len(forecast)):
        forecast[i] = alpha * demand[i - 1] + (1 - alpha) * forecast[i - 1]

    # Return the result dataframe
    df = pd.DataFrame.from_dict({'Demand': demand, 'Forecast': forecast, 'Error': demand - forecast})
    return df
    # KPIs of the forecasting technique
df = simple_exponential_smoothing(demand, fcst_period=1, alpha=0.4)
KPI(df)

# Visualize the forecasting result
df.index.name = 'Month'
plt.plot(df['Demand'], label='Demand')
plt.plot(df['Forecast'], label='Forecast')
plt.legend()
plt.title('Demand vs Simple Exponential Smoothing Forecast')
plt.xlabel('Month')
plt.ylabel('Value')

# %%
# ## 1.5. Double Exponential Smoothing
# SES chỉ xử lý level (mức trung bình)& DES xử lý cả level và trend (xu hướng)
def double_exponential_smoothing(demand, fcst_period=1, alpha=0.4, beta=0.4):
    # Create forecast, level, and trend arrays
    forecast = np.full(len(demand), np.nan)
    level = np.full(len(demand), np.nan)
    trend = np.full(len(demand), np.nan)

    # Update demand, forecast, level, and trend arrays
    demand = np.append(demand, fcst_period * np.nan)
    forecast = np.append(forecast, fcst_period * np.nan)
    level = np.append(level, fcst_period * np.nan)
    trend = np.append(trend, fcst_period * np.nan)

    # Initiate the first forecast
    level[0] = demand[0]
    trend[0] = demand[1] - demand[0]
    forecast[0] = demand[0]
    forecast[1] = level[0] + trend[0]

    # Forecast the rest
    for i in range(2, len(forecast)):
        level[i - 1] = alpha * demand[i - 1] + (1 - alpha) * (level[i - 2] + trend[i - 2])
        trend[i - 1] = beta * (level[i - 1] - level[i - 2]) + (1 - beta) * trend[i - 2]
        forecast[i] = level[i - 1] + trend[i - 1]

    # Return the result dataframe
    df = pd.DataFrame.from_dict({'Demand': demand, 'Forecast': forecast, 'Error': demand - forecast})
    return df
    # KPIs of the forecasting technique
df = double_exponential_smoothing(demand, fcst_period=1, alpha=0.4, beta=0.4)
KPI(df)

# Visualize the forecasting result
df.index.name = 'Month'
plt.plot(df['Demand'], label='Demand')
plt.plot(df['Forecast'], label='Forecast')
plt.legend()
plt.title('Demand vs Double Exponential Smoothing Forecast')
plt.xlabel('Month')
plt.ylabel('Value')

# %%
# ## 1.6. Model Optimization (Simple and Double Exponential Smoothing)
# Tối ưu alpha và beta bằng cách thử tất cả các giá trị từ 0.1-0.6 với step 0.1

def exponential_smoothing_optimization(demand, fcst_period=6):
    params = []  # alphas and betas
    KPIs = []  # result of KPIs
    dfs = []  # result dataframes

    # Loop alpha options
    for alpha in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        df = simple_exponential_smoothing(demand, fcst_period, alpha)
        params.append(f'Simple Smoothing, alpha: {alpha}')
        dfs.append(df)
        MAE = df['Error'].abs().mean()
        KPIs.append(MAE)

        # Loop beta options
        for beta in [0.05, 0.1, 0.2, 0.3, 0.4]:
            df = double_exponential_smoothing(demand, fcst_period, alpha, beta)
            params.append(f'Double Smoothing, alpha: {alpha}, beta: {beta}')
            dfs.append(df)
            MAE = df['Error'].abs().mean()
            KPIs.append(MAE)

    # Choosing the dataframe with the best KPI
    mini = np.argmin(KPIs)
    print(f'Best solution found for {params[mini]} MAE of', round(KPIs[mini], 2))
    print(KPIs)

    return dfs[mini]
# Optimization result
df = exponential_smoothing_optimization(demand)

# Visualize the forecasting result
df.index.name = 'Month'
plt.plot(df['Demand'], label='Demand')
plt.plot(df['Forecast'], label='Forecast')
plt.legend()
plt.title('Demand vs Optimized Exponential Smoothing Forecast')
plt.xlabel('Month')
plt.ylabel('Value')

# %%
# ## 1.7. Double Exponential Smoothing With Damped Trend
def double_exponential_smoothing_with_damped_trend(
    demand, fcst_period=1, alpha=0.4, beta=0.4, phi=0.9
):
    # Create forecast, level, and trend arrays
    forecast = np.full(len(demand), np.nan)
    level = np.full(len(demand), np.nan)
    trend = np.full(len(demand), np.nan)

    # Update demand, forecast, level, and trend arrays
    demand = np.append(demand, fcst_period * np.nan)
    forecast = np.append(forecast, fcst_period * np.nan)
    level = np.append(level, fcst_period * np.nan)
    trend = np.append(trend, fcst_period * np.nan)

    # Initiate the first forecast
    level[0] = demand[0]
    trend[0] = demand[1] - demand[0]
    forecast[0] = demand[0]
    forecast[1] = level[0] + trend[0]

    # Forecast the rest
    for i in range(2, len(forecast)):
        level[i - 1] = (
            alpha * demand[i - 1] + (1 - alpha) * (level[i - 2] + phi * trend[i - 2])
        )
        trend[i - 1] = beta * (level[i - 1] - level[i - 2]) + (1 - beta) * phi * trend[i - 2]
        forecast[i] = level[i - 1] + trend[i - 1]

    # Return the result dataframe
    df = pd.DataFrame.from_dict({'Demand': demand, 'Forecast': forecast, 'Error': demand - forecast})
    return df
    # KPIs of the forecasting technique
df = double_exponential_smoothing_with_damped_trend(demand, fcst_period=1, alpha=0.4, beta=0.4)
KPI(df)
# Visualize the forecasting result
df.index.name = 'Month'
plt.plot(df['Demand'], label='Demand')
plt.plot(df['Forecast'], label='Forecast')
plt.legend()
plt.title('Demand vs Double Exponential Smoothing With Damped Trend Forecast')
plt.xlabel('Month')
plt.ylabel('Value')

# %%
# ## 1.8. Triple Exponential Smoothing (Multiplicative) With Damped Trend
def triple_exponential_smoothing_with_damped_trend(
    demand, season_length=12, fcst_period=1, alpha=0.4, beta=0.4, phi=0.9, gamma=0.3
):
    # Create forecast, level, trend, and season arrays
    forecast = np.full(len(demand), np.nan)
    level = np.full(len(demand), np.nan)
    trend = np.full(len(demand), np.nan)
    season = np.full(len(demand), np.nan)

    # Calculate average monthly demand
    average_monthly_demand = demand[:].sum() / demand.shape[0]

    # Create average each month demand array
    average_each_month_demand = np.full(season_length, np.nan)

    # Calculate seasonal index for the first cycle
    for i in range(0, season_length):
        average_each_month_demand[i] = demand[i:len(demand):season_length].mean()
        season[i] = average_each_month_demand[i] / average_monthly_demand

    # Update demand, forecast, level, trend, and season arrays
    demand = np.append(demand, fcst_period * np.nan)
    forecast = np.append(forecast, fcst_period * np.nan)
    level = np.append(level, fcst_period * np.nan)
    trend = np.append(trend, fcst_period * np.nan)
    season = np.append(season, fcst_period * np.nan)

    # Forecast initiation
    forecast[0] = demand[0]
    level[0] = demand[0] / season[0]
    trend[0] = demand[1] / season[1] - demand[0] / season[0]

    # Forecast for the first cycle
    for i in range(1, season_length):
        forecast[i] = (level[i - 1] + phi * trend[i - 1]) * season[i]
        level[i] = alpha * demand[i] / season[i] + (1 - alpha) * (level[i - 1] + phi * trend[i - 1])
        trend[i] = beta * (level[i] - level[i - 1]) + (1 - beta) * phi * trend[i - 1]

    # Forecast the rest
    for i in range(season_length, len(demand)):
        forecast[i] = (level[i - 1] + phi * trend[i - 1]) * season[i - season_length]
        level[i] = alpha * demand[i] / season[i - season_length] + (1 - alpha) * (level[i - 1] + phi * trend[i - 1])
        trend[i] = beta * (level[i] - level[i - 1]) + (1 - beta) * phi * trend[i - 1]
        season[i] = gamma * demand[i] / level[i] + (1 - gamma) * season[i - season_length]

    # Return the result dataframe
    df = pd.DataFrame.from_dict({"Demand": demand, "Forecast": forecast, "Error": demand - forecast})
    return df
    # KPIs of the forecasting technique
df = triple_exponential_smoothing_with_damped_trend(
    demand, season_length=12, fcst_period=1, alpha=0.4, beta=0.4, phi=0.9, gamma=0.3
)
KPI(df)
# Visualize the forecasting result
df.index.name = 'Month'
plt.plot(df['Demand'], label='Demand')
plt.plot(df['Forecast'], label='Forecast')
plt.legend()
plt.title('Demand vs Triple Exponential Smoothing (Multiplicative) With Damped Trend Forecast')
plt.xlabel('Month')
plt.ylabel('Value')

# %%
# ## 1.9. Outliers
# Demand array create
demand_outlier = demand

# Winsorization

# Calculate the limits based on percentiles (→ 1% giá trị thấp nhất và 1% giá trị cao nhất bị xem là cực đoan)
higher_limit = np.percentile(demand_outlier, 99).astype(int)
lower_limit = np.percentile(demand_outlier, 1).astype(int)

# Print the limits
print(f'''Higher limit: {round(higher_limit, 0)}''')
print(f'''Lower limit: {round(lower_limit, 0)}''')

# Display demand array before cleaning
print(f'''Demand array before cleaning: {demand_outlier}''')

# Apply Winsorization to the demand array
# Các giá trị vượt quá giới hạn sẽ được thay thế bằng giới hạn tương ứng hay nói cách khác là:
# Nếu > upper limit → kéo xuống upper limit
# Nếu < lower limit → kéo lên lower limit
# Winsorization: Khác với xóa outlier: không loại bỏ điểm dữ liệu, mà làm “bớt cực đoan”
demand_cleaned = np.clip(demand_outlier, lower_limit, higher_limit)

# Display demand array after cleaning
print(f'''Demand array after cleaning: {demand_cleaned}''')

# Visualize the result
plt.plot(demand_outlier)
plt.plot(demand_cleaned)
plt.legend(['Demand before cleaning', 'Demand after cleaning'])

# %%
# Standard deviation

# Calculate the mean and standard deviation of the demand array
mean = demand_outlier.mean()
std = demand_outlier.std()

# Calculate the limits (inverse CDF of 99% and 1%) based on normal distribution
# Tìm giá trị x sao cho P(X < x) = 99%
from scipy.stats import norm

higher_limit = round(norm.ppf(0.99, mean, std), 0).astype(int)
lower_limit = round(norm.ppf(0.01, mean, std), 0).astype(int)

# Print the cleaning result
print(f'''Higher limit: {higher_limit}''')
print(f'''Lower limit: {lower_limit}''')
print(f'''Demand array before cleaning: {demand_outlier}''')

# Apply the clipping for cleaning
demand_cleaned = np.clip(demand_outlier, lower_limit, higher_limit)

# Print the cleaned demand array
print(f'''Demand array after cleaning: {demand_cleaned}''')

# Visualize the result
plt.plot(demand_outlier)
plt.plot(demand_cleaned)
plt.legend(['Demand before cleaning', 'Demand after cleaning'])

# %% 
# Error standard deviation

# Get the forecasting result dataframe of Triple Exponential Smoothing (Multiplicative) With Damped Trend
exp_df = df
exp_df = exp_df.iloc[:-1, :]

# Calculate the mean and standard deviation of the error array
# Kiểm tra Forecast có tạo ra error bất thường không? Sau đó tạo upper/lower limit cho Error
mean = exp_df['Error'].mean()
std = exp_df['Error'].std()

# Calculate the limits (inverse CDF of 99% and 1%) based on the normal distribution
from scipy.stats import norm

higher_limit = round(norm.ppf(0.99, mean, std), 0).astype(int)
lower_limit = round(norm.ppf(0.01, mean, std), 0).astype(int)

# Print the cleaning result for error
print(f'''Higher limit for error: {higher_limit}''')
print(f'''Lower limit for error: {lower_limit}''')

# Create arrays for the higher and lower limits
higher_limit_array = np.full(len(exp_df), higher_limit)
lower_limit_array = np.full(len(exp_df), lower_limit)

# Visualize the result
plt.plot(exp_df['Demand'])
plt.plot(exp_df['Forecast'])
plt.plot(exp_df['Error'])
plt.plot(higher_limit_array)
plt.plot(lower_limit_array)
plt.legend(['Demand', 'Forecast', 'Error', 'Higher limit array', 'Lower limit array'])

# %%
# Recalculate the outliers after removing the previous outliers

# Determine the previous outliers
previous_outliers = (exp_df['Error'] > higher_limit) | (exp_df['Error'] < lower_limit)

# Recalcualte the mean and standard deviation
# Tính lại mean và std chỉ trên dữ liệu sạch.
mean_updated = exp_df.loc[~previous_outliers, 'Error'].mean()
std_updated = exp_df.loc[~previous_outliers, 'Error'].std()

# Calculate the updated limits (inverse CDF of 99% and 1%) based on normal distribution
higher_limit_updated = round(norm.ppf(0.99, mean_updated, std_updated), 0).astype(int)
lower_limit_updated = round(norm.ppf(0.01, mean_updated, std_updated), 0).astype(int)

# Cleaning result
print(f'''Updated higher limit for error: {higher_limit_updated}''')
print(f'''Updated lower limit for error: {lower_limit_updated}''')

# Array for updated higher and lower limits
higher_limit_array_updated = np.full(len(exp_df), higher_limit_updated)
lower_limit_array_updated = np.full(len(exp_df), lower_limit_updated)

# Visualize the result
plt.plot(exp_df['Demand'])
plt.plot(exp_df['Forecast'])
plt.plot(exp_df['Error'])
plt.plot(higher_limit_array_updated)
plt.plot(lower_limit_array_updated)
plt.legend(['Demand', 'Forecast', 'Error', 'Higher limit array', 'Lower limit array'])

# %%
# ## 1.10. Triple Exponential Smoothing (Additive) With Damped Trend
def triple_exponential_smoothing_with_damped_trend_additive(
    demand, season_length=12, fcst_period=1, alpha=0.4, beta=0.4, phi=0.9, gamma=0.3
):
    # Create forecast, level, trend, and season array
    forecast = np.full(len(demand), np.nan)
    level = np.full(len(demand), np.nan)
    trend = np.full(len(demand), np.nan)
    season = np.full(len(demand), np.nan)

    # Calculate average monthly demand
    average_monthly_demand = (demand[:].sum()) / (demand.shape[0])

    # Create average each month demand array
    average_each_month_demand = np.full(season_length, np.nan)

    # Calculate seasonal index for first cycle
    # Seasonal index dạng cộng: chênh lệch giữa tháng đó và mức trung bình
    # Tính trung bình từng tháng (VD: tất cả tháng 1 qua các năm) -> So với trung bình chung --> Lấy chênh lệch → seasonal index (additive)
    # Season = Demand − Average
    for i in range(0, season_length):
        average_each_month_demand[i] = demand[i : len(demand) : season_length].mean()
        season[i] = average_each_month_demand[i] - average_monthly_demand

    # Update demand, forecast, level, and trend array
    demand = np.append(demand, fcst_period * np.nan)
    forecast = np.append(forecast, fcst_period * np.nan)
    level = np.append(level, fcst_period * np.nan)
    trend = np.append(trend, fcst_period * np.nan)
    season = np.append(season, fcst_period * np.nan)

    # Forecast initiate
    # loại seasonal effect ra trước khi tính trend
    forecast[0] = demand[0]
    level[0] = demand[0] - season[0]
    trend[0] = (demand[1] - season[1]) - (demand[0] - season[0])

    # Forecast for the first cycle
    for i in range(1, season_length):
        forecast[i] = level[i - 1] + phi * trend[i - 1] + season[i]
        level[i] = alpha * (demand[i] - season[i]) + (1 - alpha) * (level[i - 1] + phi * trend[i - 1])
        trend[i] = beta * (level[i] - level[i - 1]) + (1 - beta) * phi * trend[i - 1]

    # Forecast the rest cycle
    for i in range(season_length, len(demand)):
        forecast[i] = level[i - 1] + phi * trend[i - 1] + season[i - season_length]
        level[i] = alpha * (demand[i] - season[i - season_length]) + (1 - alpha) * (level[i - 1] + phi * trend[i - 1])
        trend[i] = beta * (level[i] - level[i - 1]) + (1 - beta) * phi * trend[i - 1]
        season[i] = gamma * (demand[i] - level[i]) + (1 - gamma) * season[i - season_length]

    # Return the result dataframe
    df = pd.DataFrame.from_dict({"Demand": demand, "Forecast": forecast, "Error": demand - forecast})
    return df
# KPIs of the forecasting technique
# Damped trend set phi = 0.9 (giảm dần ảnh hưởng của trend theo thời gian)
# Nếu không dùng damped trend thì set phi = 1 (nghĩa là Nếu không damped: Trend tiếp tục tăng vô hạn)
# Gamma = 0.3 (mức độ ảnh hưởng của seasonal effect: nếu gamma cao → seasonal effect thay đổi nhanh chóng theo thời gian, nếu gamma thấp → seasonal effect ổn định theo thời gian)
# Triple Exponential Smoothing (Additive) With Damped Trend: phù hợp với dữ liệu có cả trend và seasonal effect, trong đó seasonal effect có dạng cộng và ảnh hưởng của trend giảm dần theo thời gian.
# Triple Exponential Smoothing (Multiplicative) With Damped Trend: phù hợp với dữ liệu có cả trend và seasonal effect, trong đó seasonal effect có dạng nhân và ảnh hưởng của trend giảm dần theo thời gian.
# Lựa chọn giữa additive và multiplicative phụ thuộc vào bản chất của seasonal effect trong dữ liệu.
# Nếu seasonal effect có xu hướng thay đổi tỷ lệ với mức độ của dữ liệu (ví dụ: khi mức độ tăng lên, seasonal effect cũng tăng theo tỷ lệ), thì multiplicative là lựa chọn phù hợp. Ngược lại, nếu seasonal effect có xu hướng thay đổi một lượng cố định bất kể mức độ của dữ liệu, thì additive là lựa chọn phù hợp.

df = triple_exponential_smoothing_with_damped_trend_additive(
    demand, season_length=12, fcst_period=1, alpha=0.4, beta=0.4, phi=0.9, gamma=0.3
)
KPI(df)

# Visualize the forecasting result
df.index.name = 'Month'
plt.plot(df['Demand'], label='Demand')
plt.plot(df['Forecast'], label='Forecast')
plt.legend()
plt.title('Demand vs Triple Exponential Smoothing (Additive) With Damped Trend Forecast')
plt.xlabel('Month')
plt.ylabel('Value')
# %%
