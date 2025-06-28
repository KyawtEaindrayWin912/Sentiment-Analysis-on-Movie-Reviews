import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords once if not already downloaded
nltk.download('stopwords')

# Load raw IMDb dataset
df = pd.read_csv("../data/IMDB_Dataset.csv")

# Preprocessing function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)               # remove HTML
    text = re.sub(r'[^a-z\s]', '', text)            # remove non-alphabetic
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Apply cleaning
df['clean_review'] = df['review'].apply(clean_text)
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Select only required columns
processed_df = df[['clean_review', 'sentiment_label']]

# Save processed data
processed_df.to_csv("../data/processed_data.csv", index=False)

print("✅ Processed data saved to data/processed_data.csv")
