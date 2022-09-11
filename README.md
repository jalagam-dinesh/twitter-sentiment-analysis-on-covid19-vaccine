# twitter-sentiment-analysis-on-covid19-vaccine

# Twitter Sentiment Analysis
It is a Natural Language Processing Problem where Sentiment Analysis is done by Classifying the Positive tweets from negative tweets by machine learning models for classification, text mining, text analysis, data analysis and data visualization.

# Problem Statement
The problem statement is as follows:
The objective of this task is to extract tweets related to covid-19 vaccination from the twitter and perform Sentiment Analysis on them using Naïve Bayes Classifier, to categorize and summarize the public opinions towards covid-19 vaccine under positive and negative labels.

# Twitter data streaming
Authentication of twitter is done using Tweepy module and Twitter API. After the authentication is done, live tweets are streamed using the keywords related to the subject (covid-19 vaccine). Streamed tweets are stored in a file. 

# Twitter Preprocessing and Cleaning
The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.

# Sentiment Analysis
This is the final phase in which training and testing the Naïve Bayes Classifier is done. twitter_samples dataset from nltk package is used for training and testing the Naïve Bayes model. Finally, developed model is used on live tweets.
