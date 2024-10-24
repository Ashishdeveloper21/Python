import pandas as pd
import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Function for sentiment analysis
def analyze_sentiment(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity

# Streamlit application
st.title("Customer Feedback Analysis Tool")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(r"customer_feedback_200_rows.csv")

    # Display the data
    st.write("Data Preview:", df.head())
    
    # Check if required columns are present
    if 'Review' not in df.columns or 'Rating' not in df.columns or 'Date' not in df.columns:
        st.error("CSV must contain 'Review', 'Rating', and 'Date' columns.")
    else:
        # Sentiment analysis
        df['Sentiment'] = df['Review'].apply(analyze_sentiment)
        df['Sentiment Category'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

        # Visualizations
        st.subheader("Sentiment Distribution")
        sentiment_count = df['Sentiment Category'].value_counts()
        st.bar_chart(sentiment_count)

        # Ensure 'Date' is in datetime format (DD-MM-YYYY)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')  # Convert and handle errors
        st.write("Data after Date Conversion:", df)

        # Drop rows with NaT in 'Date' if any
        df.dropna(subset=['Date'], inplace=True)
        st.write("Data after dropping NaT rows:", df)

        # Average Rating Over Time
        st.subheader("Average Rating Over Time")
        avg_rating = df.groupby(df['Date'].dt.to_period('M'))['Rating'].mean().reset_index()
        avg_rating['Date'] = avg_rating['Date'].dt.to_timestamp()  # Convert to timestamp for Streamlit

        # Check if avg_rating is not empty
        if avg_rating.empty:
            st.error("No data available for average rating calculation.")
        else:
            st.line_chart(avg_rating.set_index('Date')['Rating'])

        # Check if Review column has any valid entries for the word cloud
        if df['Review'].notna().any() and not df['Review'].str.strip().eq('').all():
            st.subheader("Word Cloud of Reviews")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df['Review']))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.error("No valid reviews available for word cloud generation.")




