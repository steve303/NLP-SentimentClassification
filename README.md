# Sentiment Classification 
## Objective
In this project we trained a pre-built model (BERT) using the transformers library from Hugging Face on a dataset of labeled tweets, labeled for sarcasm (sarcastic/not sarcastic) See [link](data_description.md) for details on training data. We then used this model to predict the class (sarcastic/not sarcastic) of a set of provided unlabeled tweets for comparison to a competitive baseline score. We were able to beat the baseline with our model. See [link](https://steve303.github.io/NLP-SentimentClassification/Project%20Documentation/final_summary.pdf) for a full report. 
## Repository Contents
- `./Alternative Methods &  Models/`: Contains additional models that we built and trained but which were unsuccessful at beating the baseline score.
- `./data/`: Contains the test and train data provided for the competition.
- `./Project Documentation/`: Contains the final report, demo, and other project deliverables.
- `answer.txt`: Our final file containing the classification of the test tweets which outperformed the F1 score of the baseline.
- `Bert.ipynb`: Our notebook file in which we build, train, and test the model. 
- `TEXT_PREPROCESSING.py`: A dependecy of Bert.ipynb, used to preprocess the text for tokenization. 
