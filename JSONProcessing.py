import json
import os

tweets = []

# TRAIN
for line in open("./data/train.jsonl", errors = "ignore"):
    tweets.append(json.loads(line))

count = 0
for tweet in tweets:
    file_name = "tweet" + str(count) + ".txt"
    if tweet["label"] == "SARCASM":
        dir = "./train_data/sarcasm"  
    else:
        dir = "./train_data/not_sarcasm" 
    f = open(os.path.join(dir, file_name), "w+")
    f.write(tweet["response"])
    f.close()
    count += 1

# TEST
tweets = []
for line in open("./data/test.jsonl", errors = "ignore", encoding = "utf-8"):
    tweets.append(json.loads(line))

count = 0
for tweet in tweets:
    file_name = "tweet" + str(count) + ".txt"
    dir = "./test_data/data" 
    f = open(os.path.join(dir, file_name), "w+", encoding="utf-8")
    f.write(tweet["response"])
    f.close()
    count += 1
