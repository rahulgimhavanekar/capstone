import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

symbol = "ICICIBANK"

df = pd.read_csv(f"{symbol}.csv")

df[['Close']].plot()
plt.title(symbol)
plt.show()

# Cumulative Return
dr = df.cumsum()
dr.plot()
plt.title(f'{symbol} Cumulative Returns')
plt.show()

# Correlation plot
plt.figure(figsize=(10, 10))
lag_plot(df['Open'], lag=5)
plt.title('Tesla Autocorrelation plot')
plt.show()

train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
plt.figure(figsize=(12, 7))
plt.title(f'{symbol} Prices')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(df['Open'], 'blue', label='Training Data')
plt.plot(test_data['Open'], 'green', label='Testing Data')
plt.xticks(np.arange(0, 1235, 100), df['Date'][0:1235:100])
plt.legend()
plt.show()


def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true))))


train_ar = train_data['Open'].values
test_ar = test_data['Open'].values

history = [x for x in train_ar]
print(type(history))
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
    # print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test_ar, predictions)
# print('Testing Mean Squared Error: %.3f' % error)
error2 = smape_kun(test_ar, predictions)
# print('Symmetric mean absolute percentage error: %.3f' % error2)

plt.figure(figsize=(12, 7))
plt.plot(df['Open'], 'green', label='Training Data')
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed',
         label='Predicted Price')
plt.plot(test_data.index, test_data['Open'], color='red', label='Actual Price')
plt.title(f'{symbol} Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(0, 1235, 100), df['Date'][0:1235:100])
plt.legend()
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed',
         label='Predicted Price')
plt.plot(test_data.index, test_data['Open'], color='red', label='Actual Price')
plt.xticks(np.arange(1000, 1235, 20), df['Date'][1000:1235:20])
plt.title(f'{symbol} Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.legend()
plt.show()
