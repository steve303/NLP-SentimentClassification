import emoji
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
import re
import tensorflow as tf

def preprocess_text(input_data):
    # Removing tags
    #data = tf.strings.regex_replace(input_data, "@USER", " ")
    #data = tf.strings.regex_replace(data, "<URL>", " ")

    # Process emojis
    data = emoji.demojize(input_data, delimiters=(" ", " "))

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in data if not w in stop_words]

    return filtered