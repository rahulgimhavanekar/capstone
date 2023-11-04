import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Stock Price Analysis App")

symbol = st.selectbox("Select a stock symbol", [
                      "HDFCBANK", "ICICIBANK", "CANBK", "KOTAKBANK", "SBIN"])


@st.cache_data
def load_data(symbol):
    df = pd.read_csv(f"{symbol}.csv")
    return df


df = load_data(symbol)

# Plot the historical stock prices
st.subheader("Historical Stock Prices")
st.line_chart(df, x="Date", y="Close", range=["2022-07-01", "2023-07-01"])

# Calculate and plot cumulative returns
dr = df['Open'].cumsum()
st.subheader("Cumulative Returns")
st.line_chart(dr)

# Correlation plot
st.subheader("Autocorrelation Plot")
plt.figure(figsize=(10, 10))
lag_plot(df['Open'], lag=5)
st.pyplot()

# Split the data into training and testing
train_data, test_data = df[0:int(len(df) * 0.8)], df[int(len(df) * 0.8):]

# Model and Predictions
st.subheader("Price Predictions")
train_ar = train_data['Open'].values
test_ar = test_data['Open'].values
history = [x for x in train_ar]
predictions = list()


def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true))))


for t in range(len(test_ar)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)

# Display the predictions
st.line_chart(pd.DataFrame(
    {'Predicted Price': predictions, 'Actual Price': test_data['Open']}))

# Calculate and display error metrics
error_mse = mean_squared_error(test_ar, predictions)
error_smape = smape_kun(test_ar, predictions)
st.write(f"Mean Squared Error: {error_mse:.3f}")
st.write(f"Symmetric mean absolute percentage error: {error_smape:.3f}")
