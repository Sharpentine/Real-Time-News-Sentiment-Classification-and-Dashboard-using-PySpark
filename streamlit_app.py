import streamlit as st
import requests
import pandas as pd
import altair as alt
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from textblob import TextBlob
import nltk
import sys

@st.cache_resource
def initial_setup():
    try:
        nltk.download('punkt')
    except Exception as e:
        st.error(f"Error during NLTK download: {e}")

initial_setup()

st.set_page_config(
    page_title="News Sentiment Dashboard (with PySpark)",
    page_icon="🔥",
    layout="wide"
)

@st.cache_resource
def get_spark_session():
    return (
        SparkSession.builder
        .appName("StreamlitPySparkApp")
        .master("local[*]")
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )

@st.cache_data(ttl=600)
def fetch_and_analyze_news_with_spark(api_key: str, query: str) -> pd.DataFrame:
    url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={query}&language=en"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get('results', [])
        if not articles:
            st.warning("No articles found for this query.")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return pd.DataFrame()

    articles_to_process = [
        {"title": article.get("title"), "url": article.get("link")}
        for article in articles if article.get("title")
    ]
    if not articles_to_process:
        return pd.DataFrame()

    spark = get_spark_session()
    news_df = spark.createDataFrame(articles_to_process)

    def classify_sentiment(text: str) -> str:
        if not text: return 'Neutral'
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1: return 'Positive'
        if polarity < -0.1: return 'Negative'
        return 'Neutral'

    sentiment_udf = udf(classify_sentiment, StringType())
    classified_df = news_df.withColumn("sentiment", sentiment_udf(news_df["title"]))
    
    return classified_df.toPandas()

st.title("🔥 Real-Time News Sentiment Dashboard with PySpark")
st.markdown("Powered by **newsdata.io**, **PySpark**, and **Streamlit**.")

with st.sidebar:
    st.header("⚙️ Controls")
    api_key_secret = st.secrets.get("NEWSDATA_API_KEY", "")
    if not api_key_secret:
        api_key_secret = st.text_input("Enter your newsdata.io API Key", type="password")
    
    query = st.text_input("Enter a news topic (e.g., AI, Tesla, finance)", "technology")
    
    refresh_button = st.button("Fetch and Analyze News", type="primary")

if refresh_button and api_key_secret:
    with st.spinner("Starting Spark and running analysis... This may take a moment."):
        df = fetch_and_analyze_news_with_spark(api_key_secret, query)

    if not df.empty:
        st.success(f"Successfully analyzed {len(df)} articles for '{query}' using PySpark!")

        sentiment_counts = df['sentiment'].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Articles", len(df))
        col2.metric("✅ Positive", sentiment_counts.get('Positive', 0))
        col3.metric("❌ Negative", sentiment_counts.get('Negative', 0))

        st.subheader("Sentiment Distribution")
        chart_data = pd.DataFrame(sentiment_counts).reset_index()
        chart_data.columns = ['sentiment', 'count']
        
        chart = alt.Chart(chart_data).mark_bar(size=40).encode(
            x=alt.X('sentiment', axis=alt.Axis(title='Sentiment')),
            y=alt.Y('count', axis=alt.Axis(title='Number of Articles')),
            color=alt.Color('sentiment', 
                            scale=alt.Scale(
                                domain=['Positive', 'Negative', 'Neutral'],
                                range=['#2ca02c', '#d62728', '#7f7f7f']
                            )),
            tooltip=['sentiment', 'count']
        ).properties(title=f"Sentiment for '{query}'")
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Analyzed Headlines")
        st.dataframe(df, use_container_width=True, column_config={
            "url": st.column_config.LinkColumn("Article Link")
        })
    else:
        st.error("Could not retrieve or analyze news. Check your API key or query.")
else:
    st.info("Enter your newsdata.io API key and a topic, then click 'Fetch and Analyze News' to begin.")
