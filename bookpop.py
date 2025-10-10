import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
import os
import fuzzywuzzy
try:
    from fuzzywuzzy import process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

nltk.download('stopwords')

SENT_MODEL_PATH = "sent_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
SCALER_PATH = "scaler.pkl"
KMEANS_PATH = "kmeans.pkl"

df = pd.read_csv("laptops_dataset_final_600.csv")

for col in ['no_ratings','no_reviews']:
    df[col] = df[col].astype(str).str.replace(',', '').astype(int)

df = df.sample(n=5000, random_state=42)

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    ps = PorterStemmer()
    words = [ps.stem(w) for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(words)

df['clean_review'] = df['review'].fillna('').apply(preprocess)

def rating_to_sentiment(r):
    if r <= 2:
        return "negative"
    elif r == 3:
        return "neutral"
    else:
        return "positive"

df['sentiment'] = df['rating'].apply(rating_to_sentiment)

if os.path.exists(SENT_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    print("Loading trained sentiment model...")
    sent_model = pickle.load(open(SENT_MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
else:
    print("Training sentiment model...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['clean_review'])
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sent_model = LogisticRegression(solver='saga', max_iter=500)
    sent_model.fit(X_train, y_train)

    preds = sent_model.predict(X_test)
    print("\n--- Sentiment Model Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    pickle.dump(sent_model, open(SENT_MODEL_PATH, 'wb'))
    pickle.dump(vectorizer, open(VECTORIZER_PATH, 'wb'))

X_full = vectorizer.transform(df['clean_review'])
df['pred_sentiment'] = sent_model.predict(X_full)
sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}
df['sentiment_score'] = df['pred_sentiment'].map(sentiment_map)

laptop_sentiment = df.groupby('product_name', as_index=False)['sentiment_score'].mean()

laptops_df = df[['product_name','overall_rating','no_ratings','no_reviews']].drop_duplicates('product_name')
laptops_df = laptops_df.merge(laptop_sentiment, on='product_name', how='left')

features = laptops_df[['sentiment_score','no_ratings','no_reviews']].fillna(0)

if os.path.exists(SCALER_PATH) and os.path.exists(KMEANS_PATH):
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    kmeans = pickle.load(open(KMEANS_PATH, 'rb'))
    features_scaled = scaler.transform(features)
    laptops_df['cluster'] = kmeans.predict(features_scaled)
else:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=5, random_state=42)
    laptops_df['cluster'] = kmeans.fit_predict(features_scaled)

    pickle.dump(scaler, open(SCALER_PATH, 'wb'))
    pickle.dump(kmeans, open(KMEANS_PATH, 'wb'))


def show_sentiment(df, laptop_name):
    laptop_rows = df[df['product_name'].str.lower() == laptop_name.lower()]
    if laptop_rows.empty:
        print("Laptop not found in dataset.")
        return
    
    avg_sentiment = laptop_rows['sentiment_score'].mean()
    total = len(laptop_rows)
    pos = len(laptop_rows[laptop_rows['pred_sentiment']=='positive'])
    neg = len(laptop_rows[laptop_rows['pred_sentiment']=='negative'])
    neu = len(laptop_rows[laptop_rows['pred_sentiment']=='neutral'])
    
    print(f"\n--- Sentiment Analysis for '{laptop_name}' ---")
    print(f"Average Sentiment Score: {avg_sentiment:.2f}")
    print(f"Positive: {pos/total*100:.1f}% | Neutral: {neu/total*100:.1f}% | Negative: {neg/total*100:.1f}%")

def recommend_laptops(df, laptop_name):
    cluster_id = df.loc[df['product_name']==laptop_name,'cluster'].values[0]
    recs = df[df['cluster']==cluster_id].sort_values(by='sentiment_score', ascending=False)
    print(f"\n--- Recommended laptops similar to '{laptop_name}' ---")
    print(recs[['product_name','overall_rating','no_ratings','sentiment_score']].head(5))

def find_laptop(df, user_input):
    user_input = user_input.strip().lower()
    df['clean_name'] = df['product_name'].str.strip().str.lower().str.replace(r'\.+$', '', regex=True)

    if user_input in df['clean_name'].values:
        return df.loc[df['clean_name'] == user_input, 'product_name'].values[0]
    
    contains_match = df[df['clean_name'].str.contains(user_input)]
    if not contains_match.empty:
        return contains_match.iloc[0]['product_name']

    if FUZZY_AVAILABLE:
        match = process.extractOne(user_input, df['clean_name'])
        if match[1] >= 60:
            return match[0]
    
    return None

user_input = input("Enter the laptop name: ")
matched_name = find_laptop(laptops_df, user_input)
if matched_name:
    show_sentiment(df, matched_name)
    recommend_laptops(laptops_df, matched_name)
else:
    print("Laptop not found. Try typing a more complete name or check spelling.")
