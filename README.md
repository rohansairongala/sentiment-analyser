# 🎬 Movie Review Sentiment Analyser

A machine learning model that classifies IMDB movie reviews as 
positive or negative using NLP techniques.

## Overview
- **Dataset:** IMDB Movie Reviews (50,000 labelled reviews)
- **Model:** Logistic Regression with TF-IDF features
- **Accuracy:** ~89% (updated after training)
- **Demo:** [Live app on Streamlit](#) ← update this link later

## Tech Stack
Python · scikit-learn · NLTK · pandas · Streamlit

## Project Structure
\```
sentiment-analyser/
├── notebooks/   # Exploration and training
├── src/         # Reusable preprocessing functions  
├── data/        # Local only — not committed
└── README.md
\```

## How to Run Locally
\```bash
pip install -r requirements.txt
streamlit run app.py
\```

## What I Learned
- Text preprocessing: cleaning HTML, stopword removal, stemming
- Feature extraction with TF-IDF vectorisation
- Model evaluation: accuracy, F1 score, confusion matrix
- Deploying an ML model as a live web app
