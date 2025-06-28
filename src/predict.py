import re
import joblib
import nltk
from nltk.corpus import stopwords

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load trained model and vectorizer
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# 2. Preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)                # remove HTML
    text = re.sub(r"[^a-z\s]", "", text)             # remove punctuation/numbers
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# 3. Predict sentiment
def predict_sentiment(review_text):
    clean_text = preprocess_text(review_text)
    features = vectorizer.transform([clean_text])
    prediction = model.predict(features)[0]
    return "positive" if prediction == 1 else "negative"

# 4. Run sample predictions
if __name__ == "__main__":
    sample = input("Enter a movie review: ")
    result = predict_sentiment(sample)
    print(f"🔍 Sentiment Prediction: {result}")
