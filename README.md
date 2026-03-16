# 🎬 Movie Review Sentiment Analyser

A machine learning model that classifies IMDB movie reviews as positive or negative using NLP techniques.

**Live demo:** https://sentiment-analyser-rohan.streamlit.app

---

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 88.38% |
| Precision | 88.15% |
| Recall    | 88.69% |
| F1 Score  | 88.42% |

Evaluated on 25,000 held-out test reviews from the IMDB dataset.

---

## How it works

1. **Data:** 50,000 IMDB movie reviews (25k train, 25k test) — balanced 50/50 positive/negative
2. **Preprocessing:** HTML removal, lowercasing, stopword removal, Porter stemming
3. **Features:** TF-IDF vectorisation with 10,000 features and bigrams (ngram_range 1-2)
4. **Model:** Logistic Regression — outperformed Naive Bayes baseline (88.38% vs 85.12%)
5. **Deployment:** Streamlit app hosted on Streamlit Cloud

---

## Project structure
sentiment-analyser/
├── app.py                        # Streamlit web app
├── requirements.txt              # Dependencies
├── model/
│   ├── sentiment_model.pkl       # Trained Logistic Regression model
│   └── tfidf_vectoriser.pkl      # Fitted TF-IDF vectoriser
├── src/
│   └── preprocess.py             # Text cleaning pipeline
└── notebooks/
└── 01_exploration.ipynb      # Full ML workflow with outputs

---

## Key findings

- Logistic Regression outperformed Naive Bayes by 3.26% accuracy
- Bigrams significantly improve performance — captures negation context e.g. "not good"
- Model handles clear sentiment well — 99.9% confidence on strongly negative reviews
- Known limitations: sarcasm, mixed sentiment reviews, and negation phrases cause errors

---

## Tech stack

Python · scikit-learn · NLTK · pandas · NumPy · Streamlit · Hugging Face Datasets

---

## Run locally
```bash
git clone https://github.com/rohansairongala/sentiment-analyser.git
cd sentiment-analyser
pip install -r requirements.txt
streamlit run app.py
```

---

## What I learned

- Full ML workflow from raw data to deployed web application
- Text preprocessing — HTML removal, stopwords, stemming
- TF-IDF feature extraction and the importance of bigrams
- Model comparison and selection using accuracy and F1 score
- Evaluating models beyond accuracy — confusion matrix, precision, recall
- Identifying model failure modes through error analysis
- Deploying an ML model as a live web application
