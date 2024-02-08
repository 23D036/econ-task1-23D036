import pandas as 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Convert the given data into a data frame
data = pd.DataFrame({
  "date": pd.to_datetime(["2024-01-17", "2024-01-16", "2024-01-12", "2024-01-11", "2024-01-10", "2024-01-09", "2024-01-08", "2024-01-05", "2024-01-04", "2024-01-03", "2024-01-02", "2023-12-29", "2023-12-28", "2023-12-27", "2023-12-26", "2023-12-22", "2023-12-21", "2023-12-20", "2023-12-19", "2023-12-18"]),
  "open": [315.72, 315.64, 315.29, 315.47, 312.46, 311.70, 310.36, 311.71, 313.09, 312.96, 312.36, 311.50, 311.58, 309.17, 308.18, 312.01, 308.41, 311.95, 316.28, 311.64],
  "high": [317.87, 317.15, 316.34, 316.03, 315.12, 312.85, 313.02, 312.19, 314.29, 314.24, 314.48, 313.89, 311.98, 311.13, 310.95, 312.41, 310.95, 313.71, 316.31, 315.04],
  "low": [315.72, 314.08, 313.67, 312.73, 312.02, 309.44, 308.61, 307.31, 311.71, 310.53, 310.54, 311.39, 310.53, 309.17, 308.18, 308.68, 307.09, 308.82, 311.37, 311.64],
  "close": [317.03, 315.65, 316.31, 314.28, 315.04, 312.01, 312.86, 309.16, 311.75, 312.00, 311.64, 313.09, 311.07, 310.42, 310.00, 309.85, 310.55, 309.39, 312.97, 314.96],
  "volume": [446100, 499900, 692300, 624900, 418700, 629600, 525400, 492800, 739200, 720200, 800200, 499100, 311700, 415100, 377400, 330700, 484800, 696700, 1019300, 848600]
})

# Convert the data frame into a time series object
ts_data = pd.Series(data['close'].values, index=pd.to_datetime(data['date']))

# Print the time series object
print(ts_data)

# Fit the ARIMA model to the time series data
arima_model = ARIMA(ts_data, order=(1,0,0))
arima_model_fit = arima_model.fit()

# Forecast the next two weeks of data
forecast_data = arima_model_fit.forecast(steps=12)

# Print the forecasted data
print(forecast_data)
