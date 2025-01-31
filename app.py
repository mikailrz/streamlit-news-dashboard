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
            soup = BeautifulSoup(response.content, "lxml-xml")  # Explicitly use lxml-xml parser
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
text = " ".join(news_df['description'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ---- Thematic Clustering (LDA & BERT) ----
@st.cache_resource
def cluster_news(news_df):
    # TF-IDF Vectorization + LDA
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(news_df["description"])
    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    topic_distribution = lda_model.fit_transform(X)
    news_df["lda_topic"] = np.argmax(topic_distribution, axis=1)

    # BERT Embeddings + KMeans
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = bert_model.encode(news_df["description"].tolist(), show_progress_bar=False)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    news_df["bert_topic"] = kmeans.fit_predict(embeddings)

    # Topic Labels
    topic_labels = {
        0: "Global Economy",
        1: "Market Trends",
        2: "Geopolitics & Trade",
        3: "Technology & Innovation",
        4: "Energy & Commodities"
    }
    news_df["topic_label"] = news_df["bert_topic"].map(topic_labels)

    return news_df

news_df = cluster_news(news_df)

# ---- Topic Visualization ----
st.subheader("📰 News Clustering using BERT + KMeans")
fig_topic = px.bar(
    news_df['topic_label'].value_counts(),
    x=news_df['topic_label'].value_counts().index,
    y=news_df['topic_label'].value_counts().values,
    labels={'x': 'Topic', 'y': 'Article Count'},
    title="News Clustering Topics"
)
st.plotly_chart(fig_topic)

# ---- Display News Articles ----
st.subheader("🗞️ Latest News Articles")
for _, row in news_df.iterrows():
    st.markdown(f"### {row['title']}")
    st.write(row['description'])
    st.write(f"**Sentiment:** {row['sentiment_category']} | **Topic:** {row['topic_label']}")
    st.markdown(f"[🔗 Read More]({row['link']})", unsafe_allow_html=True)
    st.write("---")

st.success("🚀 News Sentiment Analysis & Clustering Completed!")

