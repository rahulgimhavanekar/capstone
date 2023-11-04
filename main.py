import streamlit as st
import pandas as pd
import numpy as np
import requests

ALPHA_VANTAGE_API_KEY = 'IFPUAXX4M2SEDV25'

ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'
ALPHA_VANTAGE_FUNCTION = 'TIME_SERIES_DAILY_ADJUSTED'


@st.cache_data(ttl=360, show_spinner=True)
def get_stock_data(ticker):
    params = {
        'function': ALPHA_VANTAGE_FUNCTION,
        'symbol': ticker,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(
        "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=RELIANCE.BSE&interval=30min&apikey=IFPUAXX4M2SEDV25")
    data = response.json()
    if 'Time Series (Daily)' in data.keys():
        df = pd.DataFrame.from_dict(
            data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(ascending=True, inplace=True)
        return df
    else:
        return None


df = get_stock_data("IBM")
st.write(df)
st.line_chart(df.drop("5. volume", axis=1)[0:10])
