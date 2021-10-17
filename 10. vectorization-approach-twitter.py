"""

VECTORIZATION APPROACH + MACHINE LEARNING PREDICTION

ALEXANDER LUND - THESIS ESCP

"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from sklearn import model_selection,svm, naive_bayes
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv("9. processed-twitter-text-thesis.csv")

df.drop(columns=["Unnamed: 0.1", "Tweet", "lexicon based positives", "lexicon based negatives", "body word count", "own polarity", "own subjectivity"], inplace=True)
df.rename(columns={"@ Date": "datetime", "Unnamed: 0": "Index", "cleaned list": "corpus", "sentiment-direction": "target"}, inplace=True)

subdf = df[["corpus", "target"]]

train_x, test_x, train_y, test_y = model_selection.train_test_split(subdf["corpus"],subdf["target"],test_size=0.3)

Encoder = LabelEncoder()
train_y = Encoder.fit_transform(train_y)
test_y = Encoder.fit_transform(test_y)


tfidf = TfidfVectorizer(max_features=2000)
tfidf.fit(subdf["corpus"])

train_x_tfidf = tfidf.transform(train_x)
test_x_tfidf = tfidf.transform(test_x)

print(tfidf.vocabulary_)

# Machine Learning Modelling

# 1. Naive Bayes
naive = naive_bayes.MultinomialNB()
naive.fit(train_x_tfidf, train_y)

predict_nb = naive.predict(test_x_tfidf)

accuracy_nb = accuracy_score(predict_nb, test_y)

print("NB Accuracy: ", accuracy_nb)
print("\n")


# 2. Support Vector Machines
svm = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="auto")
svm.fit(train_x_tfidf, train_y)

predict_svm = svm.predict(test_x_tfidf)

accuracy_svm = accuracy_score(predict_svm, test_y)

print("SVM Accuracy: ", accuracy_svm)
print("\n")

# 3. Neural Network
mlp = MLPClassifier(activation="relu", solver="adam", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

mlp.fit(train_x_tfidf, train_y)

predict_mlp = mlp.predict(test_x_tfidf)

accuracy_mlp = accuracy_score(predict_mlp, test_y)

print("MLP Accuracy: ", accuracy_mlp)