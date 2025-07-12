# 🎬 Sentiment Analysis on IMDb Movie Reviews

This project performs **sentiment classification** on movie reviews using a simple yet effective **TF-IDF + Logistic Regression** pipeline. It determines whether a review expresses a **positive** or **negative** sentiment based on the text.

The project follows clean machine learning practices with modular code, reproducible notebooks, evaluation reports, and a Streamlit demo interface.

---

## 🧠 Overview

- ✅ Dataset: IMDb 50,000 reviews (balanced)
- ✅ Preprocessing: Lowercasing, HTML & punctuation removal, stopword filtering
- ✅ Feature Engineering: TF-IDF vectorizer (`max_features=5000`)
- ✅ Model: Logistic Regression (scikit-learn)
- ✅ Evaluation: Accuracy, precision, recall, F1-score, confusion matrix
- ✅ Streamlit UI for real-time sentiment prediction

---

## 📁 Project Structure

```

Sentiment-Analysis-on-Movie-Review/
│
├── app/                     # Streamlit web app for live prediction
│   └── app.py
│
├── data/                    # Dataset and cleaned version
│   ├── IMDB\_Dataset.csv
│   └── processed\_data.csv
│
├── models/                  # Trained model and TF-IDF vectorizer
│   ├── sentiment\_model.pkl
│   └── tfidf\_vectorizer.pkl
│
├── notebooks/               # Jupyter notebooks for EDA & experimentation
│   ├── eda\_imdb.ipynb
│   └── sentiment\_analysis.ipynb
│
├── results/                 # Evaluation results
│   ├── metrics.txt
│   └── confusion\_matrix.png
│
├── src/                     # Core scripts (clean, train, predict)
│   ├── data\_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── .gitignore
├── requirements.txt
└── README.md

````

---

## 📥 Dataset

- **Source**: [Kaggle - IMDb Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- 50,000 labeled reviews (25k positive, 25k negative)
- Place it in the `data/` folder as `IMDB_Dataset.csv`

---

## ⚙️ How to Run the Project

### 1. Clone and Setup

```bash
git clone https://github.com/KyawtEaindrayWin912/Sentiment-Analysis-on-Movie-Reviews.git
cd Sentiment-Analysis-on-Movie-Review
pip install -r requirements.txt
````

### 2. Preprocess the Dataset

```bash
cd src
python preprocessing.py
```

### 3. Train the Model

```bash
python train.py
```

### 4. Evaluate Model (Save metrics and confusion matrix)

```bash
python evaluate.py
```

### 5. Predict from Terminal

```bash
python predict.py
# You will be prompted to enter a movie review
```

### 6. Run Streamlit App (optional)

```bash
cd ../app
streamlit run app.py
```

---

## 🧪 Sample Model Performance

* Accuracy: \~89%
* Evaluation: See `results/metrics.txt`
* Confusion Matrix: See `results/confusion_matrix.png`

---

## 📌 Limitations

While the model performs well on general sentiment classification, there are known limitations:

### ❌ Negation Handling

The model struggles with understanding negations like:

* “**not good**” (interpreted as positive due to the word *good*)
* “**not bad**” (interpreted as negative due to the word *bad*)

This happens because:

* TF-IDF treats each word as an isolated feature
* There’s no semantic understanding of word combinations or context

### ❌ No Word Order Awareness

The model doesn’t consider the order of words (e.g., “great plot ruined by bad acting” vs. “bad acting ruined a great plot”).

### ❌ No Deep Language Understanding

Unlike transformer models (e.g., BERT), this model doesn’t understand context, sarcasm, or subtle nuances.

---

## 📈 Future Improvements

If extended, this project could benefit from:

* ✅ Incorporating **negation-aware preprocessing** (e.g., “not good” → “not\_good”)
* ✅ Using **n-grams** in TF-IDF to capture common phrases
* ✅ Training a **transformer-based model** (like BERT) for contextual understanding
* ✅ Hyperparameter tuning and regularization

---

## 📚 Requirements

```
pandas
scikit-learn
nltk
matplotlib
seaborn
joblib
wordcloud
streamlit
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🙋 Author

**Kyawt Eaindray Win**
[GitHub](https://github.com/KyawtEaindrayWin912)

---

## 📄 License

This project is licensed under the **MIT License**.
You’re free to use, modify, and share it for personal or educational purposes.

---

## ⭐ Show Some Love

If you found this useful, please ⭐ the repository. Thanks!
