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

CUSTOM_STOPWORDS = [
    "New York Times", "Reuters", "BBC", "CNBC", "MarketWatch", "Nasdaq",
    "SCMP", "Investopedia", "Bloomberg", "Forbes", "TheGuardian", "FT",
    "WSJ", "Economist", "Business", "Markets", "Finance", "news", "article", "href", "https", 
]

text = " ".join(news_df['description'].dropna().astype(str))

# Remove stopwords
for stopword in CUSTOM_STOPWORDS:
    text = text.replace(stopword, "")

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# ---- Thematic Clustering (Dynamic LDA & BERT) ----
@st.cache_resource
def cluster_news(news_df, num_clusters=None):
    if len(news_df) < 5:
        st.warning("Not enough articles to cluster. Skipping clustering.")
        return news_df

    # Convert text into embeddings using BERT
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = bert_model.encode(news_df["description"].tolist(), show_progress_bar=False)

    # Determine optimal number of clusters (Auto mode)
    if num_clusters is None:
        max_clusters = min(10, len(news_df) // 2)
        silhouette_scores = []

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append((k, score))

        best_k = max(silhouette_scores, key=lambda x: x[1])[0]
    else:
        best_k = num_clusters

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    news_df["bert_topic"] = kmeans.fit_predict(embeddings)

    # Extract top words for each cluster
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(news_df["description"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    topic_keywords = {}
    for i in range(best_k):
        topic_indices = np.where(news_df["bert_topic"] == i)[0]
        if len(topic_indices) > 0:
            topic_tfidf = X[topic_indices].mean(axis=0)
            top_words = feature_names[np.argsort(-topic_tfidf.A1)[:3]]  # Get top 3 words
            topic_keywords[i] = " / ".join(top_words)

    # Rename topics with extracted keywords
    news_df["topic_label"] = news_df["bert_topic"].map(topic_keywords)

    return news_df


# Sidebar control for clustering
st.sidebar.subheader("🔍 Clustering Settings")
cluster_mode = st.sidebar.radio("Select Clustering Mode", ["Auto", "Manual"])
num_clusters = None

if cluster_mode == "Manual":
    num_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=min(10, len(news_df) // 2), value=5)

news_df = cluster_news(news_df, num_clusters)

# Sort topics for better readability
topic_counts = news_df['topic_label'].value_counts().reset_index()
topic_counts.columns = ['Topic', 'Article Count']
topic_counts = topic_counts.sort_values(by='Article Count', ascending=False)  # Sort by size

fig_topic = px.bar(
    topic_counts,
    x='Topic',
    y='Article Count',
    labels={'x': 'Topic Keywords', 'y': 'Article Count'},
    title=f"News Clustering with {news_df['topic_label'].nunique()} Topics"
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
