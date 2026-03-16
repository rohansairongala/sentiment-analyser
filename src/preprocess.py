import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stemmer   = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_review(text):
    text   = re.sub(r"<.*?>", " ", text)
    text   = re.sub(r"[^a-zA-Z]", " ", text)
    text   = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)
