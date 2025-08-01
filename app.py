import nltk
import os

# Create a local nltk_data directory
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Set the NLTK data path and download 'punkt'
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)



import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("imdb_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def remove_numbers(text):
    """Removes numerical values from text."""
    return re.sub(r'\d+', '', text)

def preprocess(text):
    """Clean, tokenize, remove stopwords, stem."""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(review):
    """Predicts sentiment from raw review text."""
    clean_review = remove_numbers(review)
    clean_review = preprocess(clean_review)
    review_vec = vectorizer.transform([clean_review])
    prediction = model.predict(review_vec)
    return 'positive' if prediction[0] == 1 else 'negative'

# --------- STREAMLIT UI ----------

st.set_page_config(page_title="🎬 IMDB Sentiment Analyzer", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Classifier")

user_input = st.text_area("Enter your movie review here:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"🎯 **Predicted Sentiment:** {sentiment.capitalize()}")
    else:
        st.warning("⚠️ Please enter a review to analyze.")
