"""

DATA SOURCING FILE (WEBSCRAPER)

ALEXANDER LUND - THESIS ESCP

"""

import pandas as pd
import requests
import praw
from praw.models import MoreComments, comment_forest
import pysentiment2 as ps
import demoji
from collections import OrderedDict
from collections import defaultdict
from prawcore.exceptions import Forbidden
from textblob import TextBlob
import nltk
from twitter_scraper import get_tweets
import tweepy

"""nltk.download("brown")"""

################# REDDIT SENTIMENTS SCRAPING #################


reddit = praw.Reddit(client_id="***",
                     client_secret="***",
                     redirect_uri="***",
                     password="***",
                     user_agent="***",
                     username="***",
                     )


# verify that authentication works
print(reddit.user.me())


# key subreddits scraping
targetsubreddits = ["greeninvestor", "sustainability", "climateoffensive", "greenfinance", "esginvesting"]

hotposts = OrderedDict.fromkeys(targetsubreddits,[])

dsubreddits = defaultdict(list)


for keys in hotposts:
    for submission in reddit.subreddit(keys).top(limit=60):
        dsubreddits[keys].append(submission.title)
        dsubreddits[keys].append(submission.created)


subredditdf = pd.DataFrame().from_dict(dsubreddits)

columns1 = []

for i in targetsubreddits:
    columns1.append(i)
    if columns1 == targetsubreddits:
        for i in targetsubreddits:
            columns1.append(f"{i}-timestamp")


subredditdf = pd.DataFrame(subredditdf.values.reshape(-1, len(columns1)), columns = [columns1])

order1 = []

for i in targetsubreddits:
    order1.append(i)
    order1.append(i + "-timestamp")


subredditdf = subredditdf.reindex(columns= [order1])

body1 = pd.Series(subredditdf[subredditdf.columns[::2]].values.ravel())

timestamp1 = pd.Series(subredditdf[subredditdf.columns[1::2]].values.ravel())

subreddf = pd.concat([body1.rename("body"), timestamp1.rename("timestamp")], axis = 1).dropna()

#top level comment scraping

subcomments = OrderedDict.fromkeys(targetsubreddits,[])

dsubcomments = defaultdict(list)

for keys in subcomments:
    for submission in reddit.subreddit(keys).top("all"):
        for top_level_comment in submission.comments[1:]:
            if isinstance(top_level_comment, MoreComments):
                continue
            dsubcomments[keys].append(top_level_comment.body)
            dsubcomments[keys].append(top_level_comment.created_utc)

subcommentsdf = pd.DataFrame().from_dict(dsubcomments, orient = "index")

subcommentsdf = subcommentsdf.transpose()

subcommentsdf = pd.DataFrame(subcommentsdf.values.reshape(-1, len(columns1)), columns = [columns1])

subcommentsdf = subcommentsdf.reindex(columns= [order1])

body2 = pd.Series(subcommentsdf[subcommentsdf.columns[::2]].values.ravel())

timestamp2 = pd.Series(subcommentsdf[subcommentsdf.columns[1::2]].values.ravel())

subcomdf = pd.concat([body2.rename("body"), timestamp2.rename("timestamp")], axis = 1).dropna()

# key searchwords scraping
targetsearchwords = ["esg", "sustainable investing", "responsible investment", "impact investing", "sustainable finance", "green finance", "corporate social responsibility", "esg investing", "socially responsible investing", "climate funds", "portfolio carbon footprint", "carbon finance", "positive screening", "best in class ESG", "sustainability index", "thematic investing", "triple bottom line"]

searchposts = OrderedDict.fromkeys(targetsearchwords,[])

dsearchwords = defaultdict(list)

for keys in searchposts:
    for sub in reddit.subreddit("all").search(keys, limit= 50):
        dsearchwords[keys].append(sub.title)
        dsearchwords[keys].append(sub.created)

searchwordsdf = pd.DataFrame().from_dict(dsearchwords, orient="index")

columns2 = []

for i in targetsearchwords:
    columns2.append(i)
    if columns2 == targetsearchwords:
        for i in targetsearchwords:
            columns2.append(f"{i}-timestamp")


searchwordsdf = pd.DataFrame(searchwordsdf.values.reshape(-1, len(columns2)), columns = [columns2])

order2 = []

for i in targetsearchwords:
    order2.append(i)
    order2.append(i + "-timestamp")

searchwordsdf = searchwordsdf.reindex(columns= [order2])

body3 = pd.Series(searchwordsdf[searchwordsdf.columns[::2]].values.ravel())

timestamp3 = pd.Series(searchwordsdf[searchwordsdf.columns[1::2]].values.ravel())

searchwddf = pd.concat([body3.rename("body"), timestamp3.rename("timestamp")], axis = 1).dropna()


# key searchwords comments scraping
searchcomments = OrderedDict.fromkeys(targetsearchwords,[])

dsearchcomments = defaultdict(list)

for keys in searchposts:
    for sub in reddit.subreddit("all").search(keys):
        for top_level_comment in sub.comments[1:]:
            if isinstance(top_level_comment, MoreComments):
                continue
            dsearchcomments[keys].append(top_level_comment.body)
            dsearchcomments[keys].append(top_level_comment.created_utc)

searchcommentsdf = pd.DataFrame().from_dict(dsearchcomments, orient="index")

searchcommentsdf = searchcommentsdf.transpose()

searchcommentsdf = pd.DataFrame(searchcommentsdf.values.reshape(-1, len(columns2)), columns = [columns2])

searchcommentsdf = searchcommentsdf.reindex(columns= [order2])

body4 = pd.Series(searchcommentsdf[searchcommentsdf.columns[::2]].values.ravel())

timestamp4 = pd.Series(searchcommentsdf[searchcommentsdf.columns[1::2]].values.ravel())

searchcomdf = pd.concat([body4.rename("body"), timestamp4.rename("timestamp")], axis = 1).dropna()

print("Scraping Done!")

#final concat, cleaning removed and deleted comments/posts & convert df to csv
redditdf = pd.concat((subreddf, subcomdf, searchwddf, searchcomdf), axis=0)

posts = pd.DataFrame(redditdf,columns=["body"])

indexNames = posts[(posts.body == '[removed]') | (posts.body == '[deleted]')].index

posts.drop(indexNames, inplace=True)

redditdf.to_csv("~/Desktop/reddit-sentiments.csv")

