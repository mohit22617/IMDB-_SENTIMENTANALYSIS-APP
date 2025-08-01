##%%writefile clean_utils.py##
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Ensure necessary NLTK data is downloaded (if not already)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def remove_numbers(text):
    """Removes numerical values from text."""
    return re.sub(r'\d+', '', text)

def preprocess(text):
    """
    Preprocesses text by lowercasing, removing HTML, punctuation,
    tokenizing, removing stopwords, and stemming.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)