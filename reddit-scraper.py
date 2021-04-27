import pandas as pd
from textblob import TextBlob
from prawcore.exceptions import Forbidden
import requests
import praw
import pysentiment2 as ps
import demoji
from collections import OrderedDict
from collections import defaultdict

reddit = praw.Reddit(client_id="_BcGe2c7efYCbw",
                     client_secret="TMf68K8Qiztyos5EfFtT-GQ4cFnDLQ",
                     redirect_uri="http://localhost:8080",
                     password="B4ck2Geek1N3ss",
                     user_agent="sustainable sentiments",
                     username="sustainablesentiment",
                     )

#reddit.read_only = False

print(reddit.user.me())

targetsubreddits = ["investing", "finance", "wallstreetbets"]

hotposts = OrderedDict.fromkeys(targetsubreddits,[])

d = defaultdict(list)

for keys in hotposts:
    for submission in reddit.subreddit(keys).top(limit=30):
        d[keys].append(submission.title)

print(d)


"""textualanalysis = []

for values in d:
    textualanalysis.append(values)

listToStr = ' '.join(map(str, textualanalysis))


blob = TextBlob(listToStr)

print(blob[15])"""