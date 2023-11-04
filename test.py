import requests
import pandas as pd

response = requests.get(
    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=RELIANCE.BSE&interval=30min&apikey=IFPUAXX4M2SEDV25")
data = response.json()

if "Time Series (Daily)" in data.keys():
    df = pd.DataFrame.from_dict(
        data['Time Series (Daily)'], orient='index')
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
else:
    print(False)

print(df)
