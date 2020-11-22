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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
num_words = 20000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(tweets)
train_sequences = tokenizer.texts_to_sequences(tweets)
maxlen = max([len(x) for x in train_sequences])
tokenized_tweets = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

# tokenized example
print(tweets[9])
print(tokenized_tweets[9])

# PREPARE FOR TRAINING
tweets_with_len = [[tweet, y[i], len(tweet)] for i, tweet in enumerate(tokenized_tweets)]
random.shuffle(tweets_with_len)
tweets_with_len.sort(key=lambda x: x[2])
sorted_tweet_labels = [(tweet_lab[0], tweet_lab[1]) for tweet_lab in tweets_with_len]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_tweet_labels, output_types=(tf.int32, tf.int32))

BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
next(iter(batched_dataset))

# divide into test and train
TOTAL_BATCHES = math.ceil(len(sorted_tweet_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 10
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

# BUILD MODEL
# Model hyperparameters
VOCAB_LENGTH = num_words
EMB_DIM = 200
CNN_FILTERS = 200
DNN_UNITS = 512
OUTPUT_CLASSES = 2
DROPOUT_RATE = 0.2
NB_EPOCHS = 10

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
# replaced batched_dataset with train_data
text_model.fit(batched_dataset, epochs=NB_EPOCHS)

# Predict using model
uncat_tweets = pd.read_json('./data/test.jsonl', lines = True)
un_tweets = []
uncat_data = list(uncat_tweets["response"])

for d in uncat_data:
    un_tweets.append(preprocess_text(d))

test_sequences = tokenizer.texts_to_sequences(un_tweets)
tokenized_un_tweets = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

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