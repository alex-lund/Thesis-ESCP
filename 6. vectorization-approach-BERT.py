"""

VECTORIZATION APPROACH BERT

ALEXANDER LUND - THESIS ESCP

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
import seaborn as sns
import gensim.downloader as gensim_api
from sklearn import metrics, manifold
import transformers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

reddit = pd.read_csv("4. processed-reddit-text-thesis.csv")
twitter = pd.read_csv("processed-twitter-text-thesis.csv")

twitter.drop(columns=["Unnamed: 0.1", "Tweet", "lexicon based positives", "lexicon based negatives", "body word count", "own polarity", "own subjectivity"], inplace=True)
twitter.rename(columns={"@ Date": "datetime", "Unnamed: 0": "Index", "cleaned list": "corpus", "sentiment-direction": "target"}, inplace=True)

reddit.drop(reddit[reddit["body word count"] > 80].index, inplace = True)

tweet_nlp = twitter[["corpus", "target"]]

# determination of relevant keywords for labelling
glove = gensim_api.load("glove-wiki-gigaword-300")
"""
esgpositive = nlp.most_similar(positive=["bullish", "sustainable", "investing", "esg", "positive", "impact", "green", "reduce", "emissions"],
                               negative=["bearish", "pollution", "greenwash", "negative", "destroy"],
                               topn=30)
print(esgpositive)


esgnegative = nlp.most_similar(positive=["bearish", "unsustainable", "pollute", "greenwash", "negative", "harmful"],
                                negative=["bullish"],
                                topn=30)
print(esgnegative)"""

def get_similar_words(pos_wordslist, neg_wordslist, top, nlp):
    list_out = pos_wordslist
    for tuple in nlp.most_similar(positive=pos_wordslist, negative=neg_wordslist, topn=top):
        list_out.append(tuple[0])
    return list(set(list_out))



clusters ={}

clusters["esg-bullish"] = get_similar_words(["bullish", "sustainable", "investing", "esg", "positive", "impact", "green", "reduce", "emissions"],
                               ["bearish", "pollution", "greenwash", "negative", "destroy"],
                               top=30, nlp=glove)

clusters["esg-bearish"] = get_similar_words(["bearish", "unsustainable", "pollute", "greenwash", "negative", "harmful"],
                               ["bullish"],
                               top=30, nlp=glove)

for k, v in clusters.items():
    print(k, ": ", v[0:8], "...", len(v))


### BERT tokenization

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

nlp_ = transformers.TFBertModel.from_pretrained("bert-base-uncased")

# BERT function
def utils_bert_embedding(txt, tokenizer, nlp):
    idx = tokenizer.encode(txt)
    idx = np.array(idx)[None,:]
    embedding = nlp(idx)
    X = np.array(embedding[0][0][1:-1])
    return X

## compute BERT vector for each sentence
lst_mean_vecs = [utils_bert_embedding(txt, tokenizer, nlp_).mean(0)
                 for txt in reddit["body"]]

# create the feature matrix (n x 768)
X = np.array(lst_mean_vecs)

## compute BERT vector for target labels
dic_y = {k:utils_bert_embedding(v, tokenizer, nlp_).mean(0) for k,v
         in clusters.items()}

y = list(dic_y.values())

Y = np.array(y)

## cosine similarity between sentences and target labels
similarities = np.array([metrics.pairwise.cosine_similarity(X, Y).T]).T

labels = list(dic_y.keys())

### adjustment
for i in range(len(similarities)):
    if sum(similarities[i]) == 0:
       similarities[i] = [0]*len(labels)
       similarities[i][np.random.choice(range(len(labels)))] = 1
    similarities[i] = similarities[i] / sum(similarities[i])

predicted_prob = similarities
predicted = [labels[np.argmax(pred)] for pred in predicted_prob]


bertlabels = pd.DataFrame({"BERT-label": pd.Series(predicted)})

newreddit = reddit.join(bertlabels)

newreddit.to_csv("7. bert-predictions-labels-reddit.csv")