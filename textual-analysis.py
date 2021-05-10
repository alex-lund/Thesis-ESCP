"""

TEXTUAL ANALYSIS FILE (NLP)

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


df = pd.read_csv("~/Desktop/reddit-sentiments.csv")

body = df["body"].values

textualanalysis = []

for values in body:
    textualanalysis.append(values)

listToStr = ' '.join(map(str, textualanalysis))

blob = TextBlob(listToStr)

sentiments = []

for sentence in blob.sentences:
    if sentence == "esg":
        sentiments.append(sentence)
        sentiments.append(sentence.sentiment)
