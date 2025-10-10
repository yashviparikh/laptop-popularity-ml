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

# download stopwords if not already
nltk.download('stopwords')

# ---------- STEP 1: SCRAPE REDDIT POSTS ----------
def fetch_reddit_posts(limit=100):
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

# ---------- STEP 2: CLEAN + PREPROCESS ----------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    ps = PorterStemmer()
    words = [ps.stem(w) for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(words)

# ---------- STEP 3: MOCK SENTIMENT LABELS ----------
# For demo, we’ll generate quick mock sentiment labels
# In real use, you’d label a few hundred samples manually or import a sentiment dataset
def assign_mock_sentiment(text):
    if any(w in text for w in ["love", "amazing", "great", "wonderful", "fantastic", "inspiring"]):
        return "positive"
    elif any(w in text for w in ["bad", "boring", "terrible", "waste", "hate", "awful"]):
        return "negative"
    else:
        return "neutral"

# ---------- STEP 4: TRAIN SENTIMENT MODEL ----------
def train_sentiment_model(df):
    df["clean_text"] = df["text"].apply(preprocess)
    df["sentiment"] = df["clean_text"].apply(assign_mock_sentiment)

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n--- Sentiment Model Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    return model, vectorizer, df

# ---------- STEP 5: CLUSTER BOOKS USING K-MEANS ----------
def cluster_books(df):
    sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
    df["sentiment_score"] = df["sentiment"].map(sentiment_map)
    features = df[["sentiment_score", "score", "num_comments"]].fillna(0)

    km = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = km.fit_predict(features)

    print("\n--- K-Means Cluster Centers ---")
    print(km.cluster_centers_)
    return km, df

# ---------- STEP 6: RECOMMEND BOOKS ----------
def recommend_books(df, book_title):
    if book_title not in df["title"].values:
        return "Book not found."
    cluster_id = df.loc[df["title"] == book_title, "cluster"].values[0]
    recs = df[df["cluster"] == cluster_id].sort_values(by="score", ascending=False)
    print(f"\n--- Recommended Books similar to '{book_title}' ---")
    print(recs[["title", "score", "sentiment"]].head(5))

# ---------- MAIN ----------
if __name__ == "__main__":
    print("Fetching Reddit posts...")
    df = fetch_reddit_posts(limit=100)
    print(f"Fetched {len(df)} posts.\n")

    print("Training sentiment model...")
    model, vectorizer, df = train_sentiment_model(df)

    print("Clustering books...")
    km, df = cluster_books(df)

    sample_title = df.iloc[0]["title"]
    recommend_books(df, sample_title)

    df.to_csv("reddit_books_with_sentiment.csv", index=False)
    print("\n✅ Saved full dataset as reddit_books_with_sentiment.csv")
