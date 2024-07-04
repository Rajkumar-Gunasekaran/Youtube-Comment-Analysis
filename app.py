import os
import re
import streamlit as st
import googleapiclient.discovery
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

import nltk
nltk.download('stopwords')
nltk.download('punkt')

def get_comments(video_id):
    api_key = os.getenv('YOUTUBE_API_KEY')
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    
    while request is not None:
        response = request.execute()
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
            comments.append(comment)
        
        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100
            )
        else:
            break
    
    return comments

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.strip()
    return text

def analyze_sentiment(comments):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    for comment in comments:
        vs = analyzer.polarity_scores(comment)
        if vs['compound'] >= 0.05:
            sentiments['positive'] += 1
        elif vs['compound'] <= -0.05:
            sentiments['negative'] += 1
        else:
            sentiments['neutral'] += 1
    total = len(comments)
    for sentiment in sentiments:
        sentiments[sentiment] = (sentiments[sentiment] / total) * 100
    return sentiments

def plot_sentiments(sentiment_results):
    labels = sentiment_results.keys()
    sizes = sentiment_results.values()
    colors = ['#ff9999','#66b3ff','#99ff99']
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')
    plt.title('Sentiment Analysis')
    st.pyplot(fig)

def extract_keywords(comments):
    stop_words = set(stopwords.words('english'))
    all_words = ' '.join(comments)
    all_words = clean_text(all_words)
    word_tokens = word_tokenize(all_words)
    filtered_words = [word for word in word_tokens if word not in stop_words and len(word) > 1]

    freq_dist = FreqDist(filtered_words)

    most_common = freq_dist.most_common(10)
    words, counts = zip(*most_common)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(words, counts, color='skyblue')
    plt.title('Top 10 Keywords')
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.title('Word Cloud of Comments')
    st.pyplot(fig)

st.title('YouTube Comment Analysis')

video_url = st.text_input('Enter YouTube video URL:')
if video_url:
    try:
        video_id = video_url.split('v=')[1]
        comments = get_comments(video_id)
        if comments:
            cleaned_comments = [clean_text(comment) for comment in comments]
            sentiment_results = analyze_sentiment(cleaned_comments)
            
            st.write(f"Sentiment analysis results: {sentiment_results}")
            plot_sentiments(sentiment_results)
            extract_keywords(comments)
        else:
            st.write("No comments found for this video.")
    except Exception as e:
        st.write(f"An error occurred: {e}")
