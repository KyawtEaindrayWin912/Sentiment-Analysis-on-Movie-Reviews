import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data, split_data

# 1. Load and split data
X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# 2. Vectorize using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 3. Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", round(accuracy, 4))
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Save model and vectorizer
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/sentiment_model.pkl")
joblib.dump(tfidf, "../models/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved to /models")
