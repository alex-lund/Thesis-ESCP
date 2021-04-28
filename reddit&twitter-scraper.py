"""
DATA SOURCING FILE (WEBSCRAPER)

ALEXANDER LUND e201279 - THESIS ESCP
"""

import pandas as pd
from textblob import TextBlob
from prawcore.exceptions import Forbidden
import requests
from twitter_scraper import get_tweets
import praw
from praw.models import MoreComments, comment_forest
import pysentiment2 as ps
import demoji
from collections import OrderedDict
from collections import defaultdict


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
targetsubreddits = ["investing", "finance", "wallstreetbets", "esg" ]

hotposts = OrderedDict.fromkeys(targetsubreddits,[])

dsubreddits = defaultdict(list)

for keys in hotposts:
    for submission in reddit.subreddit(keys).top(limit=100):
        dsubreddits[keys].append(submission.title)


# key search words scraping
targetsearchwords = ["esg", "sustainable investing", "responsible investment", "impact investing", "sustainable finance", "green finance", "corporate social responsibility", "esg investing", "socially responsible investing", "climate funds", "portfolio carbon footprint", "carbon finance", "positive screening", "best in class ESG", "sustainability index", "thematic investing", "triple bottom line"]

searchposts = OrderedDict.fromkeys(targetsearchwords,[])

dsearchwords = defaultdict(list)

for keys in searchposts:
    for sub in reddit.subreddit("all").search(keys):
        dsearchwords[keys].append(sub.title)




"""textualanalysis = []

for values in d:
    textualanalysis.append(values)

listToStr = ' '.join(map(str, textualanalysis))


blob = TextBlob(listToStr)

print(blob[15])"""