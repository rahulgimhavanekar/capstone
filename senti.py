import requests
from textblob import TextBlob
import streamlit as st
import pandas as pd

# Set up NewsAPI
API_KEY = "35e1e72ab74b400984b07a4c4dace67c"

# Get stock name from user
stock_name = st.text_input("Enter stock name", "RELIANCE")

# Fetch news data
url = f"https://newsapi.org/v2/everything?q={stock_name}&apiKey={API_KEY}"
response = requests.get(url)
articles = response.json()['articles']

# Perform sentiment analysis
sentiments = []
timestamps = []
news = []
sources = []
for article in articles:
    analysis = TextBlob(article['description'])
    sentiments.append(analysis.sentiment.polarity)
    timestamps.append(article['publishedAt'])
    news.append(article['description'])
    sources.append(article['source']['name'])

# Visualize sentiment analysis results
df = pd.DataFrame({'Timestamp': timestamps,
                  'Sentiment': sentiments, 'News': news, 'Source': sources})
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Display news in a table
st.table(df)
