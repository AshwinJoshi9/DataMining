import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text_string):
    text_string = text_string.lower()
    text_string = re.sub(r'[^\w\s]', '', text_string)   # Removes punctuations from test query.
    text_string = re.sub(r'\b\d+\b', '', text_string)   # Removes numbers from test query. Works for: ashwin 124. Doesn't work for: Ashwin124.
    text_string = text_string.strip()   # Strip unwanted character
    stop_words = set(stopwords.words('english'))    # English stop words.
    tokens = word_tokenize(text_string)     # Tokenized input query.
    filtered_words = [word for word in tokens if word not in stop_words]    # Removing stop words from the input test query
    return filtered_words

# Ashwin Joshi 123 as !!. ? a sa the was wehre below