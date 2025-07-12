import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)                # remove HTML
    text = re.sub(r"[^a-z\s]", "", text)             # remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Prediction
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😞"

# Streamlit UI
st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to see if it's positive or negative.")

review_input = st.text_area("📝 Enter your review here:")

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        sentiment = predict_sentiment(review_input)
        st.success(f"Predicted Sentiment: **{sentiment}**")
