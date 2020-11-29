import emoji
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
def preprocess_text(input_data):
    data = input_data 

    # Removing tags
    #data = tf.strings.regex_replace(input_data, "@USER", " ")
    #data = tf.strings.regex_replace(data, "<URL>", " ")

    # Process emojis
    data = emoji.demojize(data, delimiters=(" ", " "))

    # Remove stopwords
    # no_stop = [w for w in data if not w in stopwords.words()]
    # data = (" ").join(no_stop)
    return data