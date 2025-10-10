# ---------- IMPORTS ----------
import praw
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import numpy as np

# download stopwords if needed
nltk.download('stopwords')

# ---------- STEP 1: USER-QUERY REDDIT FETCH ----------
def fetch_book_reviews(book_name, limit=100):
    reddit = praw.Reddit(
        client_id="wp99B1cb9ruyB_O3ZqPVaQ",
        client_secret="xdXo58oEOLyknuglpPRAiYn03iSCTw",
        user_agent="bookpop by u/Careless-Stuff-3606"
    )

    posts = []
    for submission in reddit.subreddit("books").search(book_name, limit=limit):
        if not submission.selftext:
            continue
        posts.append({
            "title": submission.title,
            "text": submission.selftext,
            "score": submission.score,
            "num_comments": submission.num_comments
        })
    return pd.DataFrame(posts)

# ---------- STEP 2: GENERAL BOOK FETCH (for recommendations) ----------
def fetch_reddit_posts(limit=150):
    reddit = praw.Reddit(
        client_id="wp99B1cb9ruyB_O3ZqPVaQ",
        client_secret="xdXo58oEOLyknuglpPRAiYn03iSCTw",
        user_agent="bookpop by u/Careless-Stuff-3606"
    )

    posts = []
    for submission in reddit.subreddit("books").hot(limit=limit):
        if not submission.selftext:
            continue
        posts.append({
            "title": submission.title,
            "text": submission.selftext,
            "score": submission.score,
            "num_comments": submission.num_comments
        })
    return pd.DataFrame(posts)

# ---------- STEP 3: TEXT PREPROCESS ----------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    ps = PorterStemmer()
    words = [ps.stem(w) for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(words)

# ---------- STEP 4: MOCK SENTIMENT ----------
def assign_mock_sentiment(text):
    # simple keyword-based sentiment
    if any(w in text for w in ["love", "amazing", "great", "wonderful", "fantastic", "inspiring"]):
        return "positive"
    elif any(w in text for w in ["bad", "boring", "terrible", "waste", "hate", "awful"]):
        return "negative"
    else:
        return "neutral"

# ---------- STEP 5: MAIN PIPELINE ----------
if __name__ == "__main__":
    # Step 5a: Get user input
    book_name = input("Enter the book name: ").strip()
    print(f"\nFetching Reddit posts mentioning '{book_name}'...\n")
    book_df = fetch_book_reviews(book_name, limit=100)

    if book_df.empty:
        print("No Reddit discussions found for that book.")
        exit()

    # Step 5b: Clean and classify sentiment
    book_df["clean_text"] = book_df["text"].apply(preprocess)
    book_df["sentiment"] = book_df["clean_text"].apply(assign_mock_sentiment)

    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    avg_sentiment = book_df["sentiment"].map(sentiment_map).mean()
    print(f"Average sentiment for '{book_name}': {avg_sentiment:.2f}")

    # Step 5c: Fetch general posts for recommendation clustering
    print("\nFetching general Reddit book posts for clustering...\n")
    general_df = fetch_reddit_posts(limit=150)
    general_df["clean_text"] = general_df["text"].apply(preprocess)
    general_df["sentiment"] = general_df["clean_text"].apply(assign_mock_sentiment)
    general_df["sentiment_score"] = general_df["sentiment"].map(sentiment_map)

    features = general_df[["sentiment_score", "score", "num_comments"]].fillna(0)
    km = KMeans(n_clusters=3, random_state=42)
    general_df["cluster"] = km.fit_predict(features)

    # Step 5d: Find cluster closest to user's book sentiment
    target_cluster = np.argmin(np.abs(km.cluster_centers_[:, 0] - avg_sentiment))

    # Step 5e: Recommend top books from same cluster
    recs = general_df[general_df["cluster"] == target_cluster].sort_values(by="score", ascending=False)
    print(f"\nRecommended books similar to '{book_name}':\n")
    print(recs[["title", "score", "sentiment"]].head(5))

    # Step 5f: Save for later
    general_df.to_csv("reddit_books_with_sentiment.csv", index=False)
    print("\n✅ Saved full dataset as reddit_books_with_sentiment.csv")
