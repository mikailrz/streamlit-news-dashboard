import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import time
import random
import re
from sklearn.metrics import silhouette_score

# Configure Streamlit Page
st.set_page_config(page_title="News Sentiment & Clustering", layout="wide")

# ---- News Sources ----
NEWS_SOURCES = [
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "parser": "xml"},
    {"url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "parser": "xml"},
    {"url": "https://www.scmp.com/rss/91/feed", "parser": "xml"},
    {"url": "https://www.theguardian.com/uk/business/rss", "parser": "xml"},
    {"url": "https://www.nasdaq.com/feed/rssoutbound?category=Markets", "parser": "xml"},
    {"url": "https://www.ft.com/?format=rss", "parser": "xml"},
    {"url": "https://abcnews.go.com/abcnews/topstories", "parser": "xml"},
    {"url": "https://moxie.foxnews.com/google-publisher/latest.xml", "parser": "xml"},
    {"url": "http://rss.cnn.com/rss/cnn_topstories.rss", "parser": "xml"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", "parser": "xml"},
    {"url": "https://www.yahoo.com/news/rss", "parser": "xml"},
]

# ---- Fetch News ----
@st.cache_data
def fetch_news():
    articles = []
    for source in NEWS_SOURCES:
        try:
            time.sleep(random.uniform(1, 3))
            response = requests.get(source["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml-xml")
            for item in soup.find_all("item"):
                title = item.title.text if item.title else "No Title"
                description = item.description.text if item.description else "No Description"
                link = item.link.text if item.link else None
                articles.append({"title": title, "description": description, "link": link})
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch from {source['url']}: {e}")
    return pd.DataFrame(articles)

# Load News Data
st.sidebar.subheader("📡 Fetch Latest News")
if st.sidebar.button("🔄 Fetch News"):
    news_df = fetch_news()
    st.sidebar.success("✅ News data updated!")
else:
    news_df = fetch_news()

if news_df.empty:
    st.warning("No news articles fetched. Try clicking 'Fetch News'.")
else:
    st.success(f"✅ {len(news_df)} news articles fetched!")

# ---- Sentiment Analysis ----
@st.cache_resource
def analyze_sentiment(news_df):
    sentiment_model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
    news_df[['sentiment', 'sentiment_score']] = news_df['description'].apply(
        lambda x: pd.Series((sentiment_model(x[:512])[0]['label'], sentiment_model(x[:512])[0]['score']))
    )
    return news_df

news_df = analyze_sentiment(news_df)

# Sentiment Mapping
sentiment_mapping = {
    '1 star': 'Negative',
    '2 stars': 'Negative',
    '3 stars': 'Neutral',
    '4 stars': 'Positive',
    '5 stars': 'Positive'
}
news_df['sentiment_category'] = news_df['sentiment'].map(sentiment_mapping)

# ---- Sentiment Visualization ----
st.subheader("📊 Sentiment Distribution")
fig_sentiment = px.bar(
    news_df['sentiment_category'].value_counts(),
    x=news_df['sentiment_category'].value_counts().index,
    y=news_df['sentiment_category'].value_counts().values,
    labels={'x': 'Sentiment Category', 'y': 'Count'},
    title="Sentiment Distribution (Positive, Neutral, Negative)"
)
st.plotly_chart(fig_sentiment)

# ---- Word Cloud ----
st.subheader("☁️ Word Cloud of News Descriptions")

CUSTOM_STOPWORDS = set([
    "New York Times", "Reuters", "BBC", "CNBC", "MarketWatch", "Nasdaq",
    "SCMP", "Investopedia", "Bloomberg", "Forbes", "The Guardian", "FT",
    "WSJ", "Economist", "Business", "Markets", "Finance", "news", "article",
    "href", "https", "http", "www", "com", "html", "utm", "rss", "feed",
    "p", "s", "u", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
    "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
])

# Combine all descriptions
text = " ".join(news_df['description'].dropna().astype(str))

# Remove stopwords, numbers, punctuation & short words (< 3 letters)
cleaned_text = " ".join([
    word for word in re.sub(r"[^\w\s]", "", text).split()  # Remove punctuation
    if word.lower() not in CUSTOM_STOPWORDS and len(word) > 2  # Remove stopwords & short words
])

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(cleaned_text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ---- Display News Articles ----
st.subheader("🗞️ Latest News Articles")
for _, row in news_df.iterrows():
    st.markdown(f"### {row['title']}")
    st.write(row['description'])
    st.write(f"**Sentiment:** {row['sentiment_category']}")
    st.markdown(f"[🔗 Read More]({row['link']})", unsafe_allow_html=True)
    st.write("---")

st.success("🚀 News Sentiment Analysis Completed!")
