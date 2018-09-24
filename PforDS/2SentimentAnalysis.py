import tweepy
from textblob import TextBlob

consumer_key = 'z1jezlmUKgce8YEIGAuYHbXPT'
consumer_secret = '0zKge1ZUfcgXQS2KANQToO3M6k1JSmL0ctDt3ovylFtoObXmK9'

access_token = '730008571-3cMR7mHqhlBMmfxBNFJsTarCL9ebuxC7LPdwLK7P'
access_token_secret = 'WsgGOp8NXaZN9v5xMPW9UmBQqQXyTXVyvG1Fp0sdCgFWu'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')

for tweet in public_tweets:
	print(tweet.text)
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)