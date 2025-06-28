import joblib
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load model + vectorizer
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")

# Load data
df = pd.read_csv("../data/processed_data.csv")
X = df['clean_review']
y = df['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# Save metrics
os.makedirs("../results", exist_ok=True)
report = classification_report(y_test, y_pred)

with open("../results/metrics.txt", "w") as f:
    f.write("Model Evaluation Report\n")
    f.write("=======================\n\n")
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png")
plt.close()
