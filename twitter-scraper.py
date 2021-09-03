"""

DATA SOURCING FILE (TWITTER WEBSCRAPER)

ALEXANDER LUND - THESIS ESCP

"""

import pandas as pd
import tweepy as tw
from collections import OrderedDict
from collections import defaultdict

consumer_key = "***********************"
consumer_secret = "***********************"
access_token = "***********************"
access_token_secret = "***********************"

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


searchwords = ["esg", "sustainable investing", "responsible investment", "impact investing", "sustainable finance", "green finance", "corporate social responsibility", "esg investing", "socially responsible investing", "climate funds", "portfolio carbon footprint", "carbon finance", "positive screening", "best in class ESG", "sustainability index", "thematic investing", "triple bottom line", "greenwashing"]
shortsearchwords = ["esg", "sustainable investing", "responsible investment"]


"""newsearch = []

for word in searchwords:
    newsearch.append(word + " -filter:retweets")"""

date_since = "2020-05-19"

shortsearch = OrderedDict.fromkeys(shortsearchwords, [])
tweets = defaultdict(list)
date =

numberoftweets = 5

for key in shortsearch:
    for tweet in tw.Cursor(api.search, q=key, lang="en", since=date_since).items(numberoftweets):
        tweets.append(tweet.text)
        date.append(tweet.date)

