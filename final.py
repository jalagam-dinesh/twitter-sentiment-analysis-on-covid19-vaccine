import os
import tweepy as tw
import pandas as pd

consumer_key= '6jDOobD4csWk7zWe1oL1RlEs6'
consumer_secret= 'IQU8vhvHQiRZekY52HVVdZacIHcikrLwk4mDErY8zQyxZPthKZ'
access_token= '1439969123476279305-Gt9s6mp9xIEPZSdphV0OFBwcTJtOjO'
access_token_secret= 'eojCqoYRwTvNvViMoPypaNWXgfqZfz6fYFzWGUuszBXle'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define the search term and the date_since date as variables
date_since = "2020-01-01"


new_search = "#covid-19 vaccine" + " -filter:retweets"

# Collect tweets
tweets = tw.Cursor(api.search_tweets,
              q=new_search,
              lang="en").items(10)

all_tweets = [tweet.text for tweet in tweets]
tweets=all_tweets[:10]


import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords

import numpy as np

nltk.download('twitter_samples')
nltk.download('stopwords')

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')



import re
import string

from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from matplotlib import pyplot as plt

def process_tweet(tweet):
  stemmer = PorterStemmer() 
  stopwords_english = stopwords.words('english')

  # remove the stock market tickers
  tweet = re.sub(r'\$\w*', '', tweet)

  # remove the old styles retweet text 'RT'
  tweet = re.sub(r'^RT[\s]+', '', tweet)

  # remove the hyperlinks
  tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
  
  # remove the # symbol
  tweet = re.sub("@[A-Za-z0-9_]+", "", tweet)
  tweet = re.sub("#[A-Za-z0-9_]+", "", tweet)
  tweet = re.sub(r'#', '', tweet)

  # Tokenize the tweet
  tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
  tweet_tokens = tokenizer.tokenize(tweet)

  tweet_clean = []

  # removing stopwords and punctuation
  for word in tweet_tokens:
    if (word not in stopwords_english and
        word not in string.punctuation):
      stem_word = stemmer.stem(word)    #stemming
      tweet_clean.append(stem_word)

  return tweet_clean




def count_tweets(tweets, ys):
  ys_list = np.squeeze(ys).tolist()
  freqs ={}

  for y, tweet in zip(ys_list, tweets):
    for word in process_tweet(tweet):
      pair = (word, y)
      if pair in freqs:
        freqs[pair] +=1
      else:
        freqs[pair] = 1
  
  return freqs


def lookup(freqs, word, label):
  n = 0
  pair = (word, label)
  if pair in freqs:
    n = freqs[pair]
  return n


# splitting the data for training and testing 
train_pos = all_positive_tweets[:4000]
test_pos = all_positive_tweets[4000:]

train_neg = all_negative_tweets[:4000]
test_neg = all_negative_tweets[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# numpy array for the labels in the training set
train_y = np.append(np.ones((len(train_pos))), np.zeros((len(train_neg))))
test_y = np.append(np.ones((len(test_neg))), np.zeros((len(test_neg))))



# Build a frequency dictionary
freqs = count_tweets(train_x, train_y)

def train_naive_bayes(freqs, train_x, train_y):
  logliklihood = {}
  logprior = 0

  # calculate V, number of unique words in the vocabulary
  vocab = set([pair[0] for pair in freqs.keys()])
  V = len(vocab)

  ## Calculate N_pos, N_neg, V_pos, V_neg
  # N_pos : total number of positive words
  # N_neg : total number of negative words
  # V_pos : total number of unique positive words
  # V_neg : total number of unique negative words

  N_pos = N_neg = V_pos = V_neg = 0
  for pair in freqs.keys():
    if pair[1]>0:
      V_pos +=1
      N_pos += freqs[pair]
    else:
      V_neg +=1
      N_neg += freqs[pair]

  # Number of Documents (tweets)
  D = len(train_y)

  # D_pos, number of positive documnets
  D_pos = len(list(filter(lambda x: x>0, train_y)))

  # D_pos, number of negative documnets
  D_neg = len(list(filter(lambda x: x<=0, train_y)))

  # calculate the logprior
  logprior = np.log(D_pos) - np.log(D_neg)

  for word in vocab:
    freqs_pos = lookup(freqs, word, 1)
    freqs_neg = lookup(freqs, word, 0)

    # calculte the probability of each word being positive and negative
    p_w_pos = (freqs_pos+1)/(N_pos+V)
    p_w_neg = (freqs_neg+1)/(N_neg+V)

    logliklihood[word] = np.log(p_w_pos/p_w_neg)
  
  return logprior, logliklihood



logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
#print(logprior)
#print(len(loglikelihood))



def naive_bayes_predict(tweet, logprior, loglikelihood):
  word_l = process_tweet(tweet)
  p = 0
  p+=logprior

  for word in word_l:
    if word in loglikelihood:
      p+=loglikelihood[word]

  return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
  accuracy = 0
  y_hats = []
  for tweet in test_x:
    if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
      y_hat_i = 1
    else:
      y_hat_i = 0
    y_hats.append(y_hat_i)
  error = np.mean(np.absolute(test_y - y_hats))
  accuracy = 1-error

  return (accuracy * 100)
  
#print("Naive Bayes accuracy = %0.2f" % (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

#tweets = ['@dinesh #dinesh I am happy https://t.co/0DlGChTBIx']
final = []
for tweet in tweets:
    p = naive_bayes_predict(tweet, logprior, loglikelihood)

    parsed_tweet = {}
    parsed_tweet['text'] = tweet
    parsed_tweet['sentiment'] = p
    final.append(parsed_tweet)
    '''if p>0:
        print('\033')
        print(f'{tweet} :: Positive sentiment ({p:.2f})')
    else:
        print('\033')
        print(f'{tweet} :: Negative sentiment ({p:.2f})')'''

#print(len(final))
# picking positive tweets from tweets
ptweets=[]
ntweets=[]  
for tweet in final:
    if tweet['sentiment']>0:
        ptweets.append(tweet)
    else:
        ntweets.append(tweet)
print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(final)))
print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(final)))

x=format(100*len(ptweets)/len(final))
y=format(100*len(ntweets)/len(final))

sentiment = ['positive','negative']
data = [x,y]
fig = plt.figure(figsize =(10, 7))
plt.pie(data, labels = sentiment)
 
# show plot
plt.show()

print("\n\nPositive tweets:")
for tweet in ptweets[:5]:
    print(tweet['text'])
  
# printing first 5 negative tweets
print("\n\nNegative tweets:")
for tweet in ntweets[:5]:
    print(tweet['text'])
    
