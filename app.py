import streamlit as st
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocess import clean_review

st.set_page_config(
    page_title="Movie Sentiment Analyser",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_model():
    model     = joblib.load("model/sentiment_model.pkl")
    vectoriser = joblib.load("model/tfidf_vectoriser.pkl")
    return model, vectoriser

model, vectoriser = load_model()

st.title("🎬 Movie Review Sentiment Analyser")
st.markdown("Type a movie review below and the model will classify it as positive or negative.")
st.markdown("---")

review = st.text_area(
    "Your review",
    placeholder="e.g. This film was absolutely brilliant, I loved every second of it...",
    height=150
)

if st.button("Analyse sentiment", type="primary"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned     = clean_review(review)
        vector      = vectoriser.transform([cleaned])
        prediction  = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]
        confidence  = max(probability) * 100

        st.markdown("---")

        if prediction == 1:
            st.success(f"POSITIVE  —  {confidence:.1f}% confidence")
            st.markdown("The model thinks this review is **positive**.")
        else:
            st.error(f"NEGATIVE  —  {confidence:.1f}% confidence")
            st.markdown("The model thinks this review is **negative**.")

        with st.expander("How does this work?"):
            st.markdown(f"""
**What the model saw after cleaning:**
`{cleaned[:300]}...`

**How it works:**
1. Your review is cleaned — HTML removed, lowercased, stopwords stripped, words stemmed
2. TF-IDF converts the cleaned text into a vector of 10,000 numbers
3. Logistic Regression predicts sentiment from that vector
4. Confidence score comes from the model\'s predicted probability

**Model accuracy:** 88.38% on 25,000 test reviews
            """)

st.markdown("---")
st.markdown(
    "Built with scikit-learn · TF-IDF · Logistic Regression · "
    "[View on GitHub](https://github.com/rohansairongala/sentiment-analyser)",
    unsafe_allow_html=True
)
