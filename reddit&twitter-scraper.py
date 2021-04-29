"""

DATA SOURCING FILE (WEBSCRAPER)

ALEXANDER LUND e201279 - THESIS ESCP

"""

import pandas as pd
from textblob import TextBlob
import nltk
from prawcore.exceptions import Forbidden
import requests
from twitter_scraper import get_tweets
import praw
from praw.models import MoreComments, comment_forest
import pysentiment2 as ps
import demoji
from collections import OrderedDict
from collections import defaultdict

"""nltk.download("brown")"""

################# REDDIT #################


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
targetsubreddits = ["investing", "finance", "wallstreetbets", "esg"]

hotposts = OrderedDict.fromkeys(targetsubreddits,[[]])

dsubreddits = defaultdict(list)


for keys in hotposts:
    for submission in reddit.subreddit(keys).top(limit=100):
        dsubreddits[keys].append(submission.title)
        dsubreddits[keys].append(submission.created)


subredditdf = pd.DataFrame().from_dict(dsubreddits)

subredditdf = pd.DataFrame(subredditdf.values.reshape(-1, 8), columns = ["investing", "finance", "wallstreetbets", "esg", "investing-timestamp", "finance-timestamp", "wallstreetbets-timestamp", "esg-timestamp"])

order1 = ["investing", "investing-timestamp", "finance", "finance-timestamp", "wallstreetbets", "wallstreetbets-timestamp", "esg", "esg-timestamp"]

subredditdf = subredditdf.reindex(columns= order1)

"""
# key search words scraping
targetsearchwords = ["esg", "sustainable investing", "responsible investment", "impact investing", "sustainable finance", "green finance", "corporate social responsibility", "esg investing", "socially responsible investing", "climate funds", "portfolio carbon footprint", "carbon finance", "positive screening", "best in class ESG", "sustainability index", "thematic investing", "triple bottom line"]

searchposts = OrderedDict.fromkeys(targetsearchwords,[])

dsearchwords = defaultdict(list)

for keys in searchposts:
    for sub in reddit.subreddit("all").search(keys):
        dsearchwords[keys].append(sub.title)
        dsearchwords[keys].append(sub.created)


searchcomments = OrderedDict.fromkeys(targetsearchwords,[])

dsearchcomments = defaultdict(list)

for keys in searchposts:
    for sub in reddit.subreddit("all").search(keys):
        for comments in reddit.subreddit("all").comments():
            dsearchcomments[keys].append(sub.comments)
            dsearchcomments[keys].append(sub.created)
"""

"""textualanalysis = []

for values in dsubreddits.values():
    textualanalysis.append(values)

for values in dsearchwords.values():
    textualanalysis.append(values)"""

"""listToStr = ' '.join(map(str, textualanalysis))

blob = TextBlob(textualanalysis)

print(blob.word_counts)"""

