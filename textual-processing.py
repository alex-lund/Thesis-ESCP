"""

TEXTUAL PROCESSING FILE (NLP)

ALEXANDER LUND - THESIS ESCP

"""
import pandas as pd
from nltk.corpus import stopwords
import spacy
import pysentiment2 as ps
import demoji
from collections import OrderedDict
from collections import defaultdict
from textblob import TextBlob


df = pd.read_csv("~/Desktop/reddit-sentiments.csv")

stopw = stopwords.words("english")

df["body"] = df["body"].astype(str)

df = df.drop(df.index[[3009]])

f"""or row in df[2942:3791]:
    if str(df["body"]).startswith("1"):
        df["body"][row].replace(to_replace=df["body"][row].values, value=df["timestamp"][row].values, inplace=True)
        df["timestamp"][row].replace(to_replace=df["timestamp"][row].values, value=df["body"][row].values, inplace=True)
"""

df.loc[df[2942:3791],'body','timestamp'] = df.loc[df[2942:3791],'timestamp','body'].values


"""df["cleaned body"] = df["body"].apply(lambda x: " ".join([word for word in x.split() if word.lower() not in (stopw)]))

df["cleaned body"] = df["cleaned body"].str.replace('[^\w\s]','')

df["cleaned body"] = df["cleaned body"].apply(lambda x: " ".join(x.lower() for x in x.split()))

df = df[df["cleaned body"].str.contains("removed|deleted")==False]

nlp = spacy.load("en_core_web_sm")

def space(comment):
    doc = nlp(comment)
    return " ".join([token.lemma_ for token in doc])

df["cleaned body"] = df["cleaned body"].apply(space)


sustainable_dictionary = ["esg", "environmental", "social", "governance", "negative impact", "planet", "positive impact", "index", "sustainability", "climate", "climate change", "green", "bond", "sustainable investing", "responsible investment", "impact investing", "sustainable finance", "green finance", "corporate social responsibility", "esg investing", "socially responsible investing", "climate funds", "portfolio carbon footprint", "carbon finance", "positive screening", "best in class ESG", "sustainability index", "thematic investing", "triple bottom line", "greenwashing", "solar", "wind", "electric", "farm", "carbon", "co2", "emissions", "investment", "energy", "crisis", "paris agreement", "renewable", "fossil fuel", "alternative", "green portfolio", "natural", "ocean", "reform", "meat consumption", "cement", "future", "tree", "deforestation", "harvest", "land abuse", "nature", "commodity", "greenwash", "oil", "environmentalist"]

sustainable_pattern = '|'.join(sustainable_dictionary)

df["esg relevant"] = df["cleaned body"].str.contains(sustainable_pattern)

df["esg relevant"] = df["esg relevant"].map({True: 1, False: 0})


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


df["Subjectivity"] =  df["cleaned body"].apply(getSubjectivity)
df["Polarity"] = df["cleaned body"].apply(getPolarity)

df["ESG relevant Polarity"] = df[["esg relevant"]].multiply(df["Polarity"], axis = "index")
"""


#df.to_csv("~/Desktop/processed-text-thesis.csv")


