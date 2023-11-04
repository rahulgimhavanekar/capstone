import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# Set up Alphavantage API
API_KEY = 'KPFL0QDXT46Y6S3H'
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Fetch stock data
data, _ = ts.get_daily(symbol='MSFT', outputsize='full')

# Preprocess data
data = data['4. close'].resample('D').mean()  # Resample data to daily
data = data.fillna(data.bfill())  # Fill any gaps in data

# Split data into training and testing sets
train_data, test_data = data[0:int(len(data)*0.8)], data[int(len(data)*0.8):]

# Create an ARIMA model
model = ARIMA(train_data, order=(5, 1, 0))
# Fit the model to the training data
model_fit = model.fit()

# Predict stock prices
forecast, stderr = model_fit.forecast(steps=len(test_data))

# Plot actual and predicted values
st.line_chart(pd.DataFrame(
    {'Actual': test_data, 'Predicted': forecast}, index=test_data.index))
