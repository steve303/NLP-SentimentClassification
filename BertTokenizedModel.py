import tensorflow as tf 
import tensorflow_hub as hub 
from tensorflow.keras import layers
import bert
import numpy as np 
import pandas as pd 
import json
import re
import random
import math
from TEXT_MODEL import TEXT_MODEL
from TEXT_PREPROCESSING import preprocess_text

# LOADING DATA
categorized_tweets = pd.read_json('./data/train.jsonl', lines = True)
categorized_tweets.isnull().values.any()
print(categorized_tweets)

# PREPROCESSING DATA
tweets = []
data = list(categorized_tweets["response"])
print(data[0])
for d in data:
    tweets.append(preprocess_text(d))

y = categorized_tweets["label"]
y = np.array(list(map(lambda x: 1 if x=="SARCASM" else 0, y)))

# TOKENIZING DATA
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3", trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

def tokenize_tweets(data):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data))

tokenized_tweets = [tokenize_tweets(tweet) for tweet in tweets]

# tokenized example
print(tweets[9])
print(tokenizer.tokenize(tweets[9]))

# PREPARE FOR TRAINING
tweets_with_len = [[tweet, y[i], len(tweet)] for i, tweet in enumerate(tokenized_tweets)]
random.shuffle(tweets_with_len)
tweets_with_len.sort(key=lambda x: x[2])
sorted_tweet_labels = [(tweet_lab[0], tweet_lab[1]) for tweet_lab in tweets_with_len]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_tweet_labels, output_types=(tf.int32, tf.int32))

BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
next(iter(batched_dataset))
TOTAL_BATCHES = math.ceil(len(sorted_tweet_labels) / BATCH_SIZE)
batched_dataset.shuffle(TOTAL_BATCHES)


# BUILD MODEL
# Model hyperparameters
VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 2
DROPOUT_RATE = 0.2
NB_EPOCHS = 20

# Model object
text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)

# Compile model
text_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit model
text_model.fit(batched_dataset, epochs=NB_EPOCHS)

# Predict using model
uncat_tweets = pd.read_json('./data/test.jsonl', lines = True)
un_tweets = []
uncat_data = list(uncat_tweets["response"])

for d in uncat_data:
    un_tweets.append(preprocess_text(d))
tokenized_un_tweets = [tokenize_tweets(tweet) for tweet in un_tweets]
print(str(len(un_tweets)))

processed_dataset = tf.data.Dataset.from_generator(lambda:tokenized_un_tweets, output_types=tf.int32)
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=(None, ))

predictions = text_model.predict(batched_dataset)

with open('answer.txt', 'w') as f:
    c = 1
    s_c = 0
    ns_c = 0
    for p in predictions:
        if p >= .5:
            f.write("twitter_" + str(c) + "," + "SARCASM\n")
            c += 1
            s_c += 1
        else:
            f.write("twitter_" + str(c) + "," + "NOT_SARCASM\n")
            c += 1
            ns_c += 1
print("# sarcasm: " + str(s_c))
print("# not sarcasm: " + str(ns_c))
