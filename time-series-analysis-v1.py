"""

TIME SERIES ANALYSIS FILE

ALEXANDER LUND - THESIS ESCP

"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from stargazer.stargazer import Stargazer
import plotly.offline
import quandl
import yfinance
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from scipy.stats import ttest_rel as ttest


# Time series analysis on sentiments

NLPdf = pd.read_csv("processed-reddit-text-thesis.csv")

NLPdf.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "cleaned body", "own subjectivity"], inplace=True)

NLPdf["datetime"] = pd.to_datetime(NLPdf["datetime"]).dt.date

NLPdfnew = NLPdf.groupby("datetime").mean()

NLPdfnew.reset_index(inplace=True)

print(NLPdfnew["own polarity"].value_counts())

graph = px.line(NLPdfnew,x="datetime",y="own polarity")

#graph.show()

# fund information
InvescoESG = yfinance.Ticker("ESG")

invescoESGdf = InvescoESG.history(start='2010-1-1', end='2021-1-1')

invescoESGdf = invescoESGdf.drop(columns=["Volume", "Dividends", "Stock Splits"])

invescoESGdf = invescoESGdf.reset_index()

for i in ['Open', 'High', 'Close', 'Low']:
      invescoESGdf[i]  =  invescoESGdf[i].astype('float64')

invescoESGdf["Delta"] = invescoESGdf["Close"] - invescoESGdf["Open"]


sin = yfinance.Ticker("VICE")

sin = sin.history(start='2010-1-1', end='2021-1-1')

sin = sin.drop(columns=["Volume", "Dividends", "Stock Splits"])

sin = sin.reset_index()

for i in ['Open', 'High', 'Close', 'Low']:
      sin[i]  =  sin[i].astype('float64')

sin["Delta"] = sin["Close"] - sin["Open"]

#
scaler = StandardScaler()

X = invescoESGdf["Delta"].values.reshape(-1,1)
scaler.fit(X)
X_scaled = scaler.transform(X)
invescoESGdf["standard_delta"] = X_scaled

scaler2 = MinMaxScaler()

X2 = invescoESGdf["standard_delta"].values.reshape(-1,1)
scaler2.fit(X2)
X2_scaled = scaler2.transform(X2)
invescoESGdf["minmax_std_delta"] = X2_scaled

"""fig = go.Figure(data=[go.Candlestick(x=invescoESGdf["Date"],
                                     open= invescoESGdf["Open"],
                                     high= invescoESGdf["High"],
                                     low= invescoESGdf["Low"],
                                     close= invescoESGdf["Close"])])

fig.show()"""


ESGdf = invescoESGdf.drop(columns=["Open", "High", "Low", "Close"])

ESGdf["Date"] = pd.to_datetime(ESGdf["Date"]).dt.date
ESGdf.rename(columns={"Date": "datetime"}, inplace=True)

SINdf = sin.drop(columns=["Open", "High", "Low", "Close"])

SINdf["Date"] = pd.to_datetime(SINdf["Date"]).dt.date
SINdf.rename(columns={"Date": "datetime"}, inplace=True)


ESG = NLPdf.merge(ESGdf, on="datetime")
SIN = NLPdf.merge(SINdf, on="datetime")

ESG = ESG.groupby("datetime").mean()
ESG.reset_index(inplace=True)

ESG = ESG.groupby("datetime").mean()
ESG.reset_index(inplace=True)

#ESG = ESG[(ESG['datetime'] < '2020-02-01') & (ESG['datetime'] < '2020-06-01')]

print("Correlation between polarity and ESG delta on intraday basis:",ESG["own polarity"].corr(ESG["Delta"]))

a = ESG["own polarity"]
b = ESG["Delta"]

ESGttest = ttest(a, b)

print(ESGttest)

print("Correlation between polarity and SIN delta on intraday basis:",SIN["own polarity"].corr(SIN["Delta"]))

c = SIN["own polarity"]
d = SIN["Delta"]

SINttest = ttest(c, d)

print(SINttest)
print("\n")

ESGlag1 = ESG
ESGlag1[["Delta", "standard_delta", "minmax_std_delta"]] = ESG[["Delta", "standard_delta", "minmax_std_delta"]].shift(1)
ESGlag1 = ESGlag1.dropna()
print("Correlation between polarity and ESG delta with 1 day lag:", ESGlag1["own polarity"].corr(ESGlag1["Delta"]))

a1 = ESGlag1["own polarity"]
b1 = ESGlag1["Delta"]

ESGlag1ttest = ttest(a1, b1)

print(ESGlag1ttest)

SINlag1 = SIN
SINlag1[["Delta"]] = SINlag1[["Delta"]].shift(1)
SINlag1 = SINlag1.dropna()
print("Correlation between polarity and SIN delta with 1 day lag:", SINlag1["own polarity"].corr(SINlag1["Delta"]))

c1 = SINlag1["own polarity"]
d1 = SINlag1["Delta"]

SINlag1ttest = ttest(c1, d1)

print(SINlag1ttest)
print("\n")

ESGlag2 = ESG
ESGlag2[["Delta", "standard_delta", "minmax_std_delta"]] = ESG[["Delta", "standard_delta", "minmax_std_delta"]].shift(15)
ESGlag2 = ESGlag2.dropna()
print("Correlation between polarity and ESG delta with 15 day lag:", ESGlag2["own polarity"].corr(ESGlag2["Delta"]))

a2 = ESGlag2["own polarity"]
b2 = ESGlag2["Delta"]

ESGlag2ttest = ttest(a2, b2)

print(ESGlag2ttest)

SINlag2 = SIN
SINlag2[["Delta"]] = SINlag2[["Delta"]].shift(15)
SINlag2 = SINlag2.dropna()
print("Correlation between polarity and SIN delta with 15 day lag:", SINlag2["own polarity"].corr(SINlag2["Delta"]))

c2 = SINlag2["own polarity"]
d2 = SINlag2["Delta"]

SINlag2ttest = ttest(c2, d2)

print(SINlag2ttest)
print("\n")

ESGlag3 = ESG
ESGlag3[["Delta", "standard_delta", "minmax_std_delta"]] = ESG[["Delta", "standard_delta", "minmax_std_delta"]].shift(30)
ESGlag3 = ESGlag3.dropna()
print("Correlation between polarity and ESG delta with 30 day lag:", ESGlag3["own polarity"].corr(ESGlag3["Delta"]))

a3 = ESGlag3["own polarity"]
b3 = ESGlag3["Delta"]

ESGlag3ttest = ttest(a3, b3)

print(ESGlag3ttest)

SINlag3 = SIN
SINlag3[["Delta"]] = SINlag3[["Delta"]].shift(30)
SINlag3 = SINlag3.dropna()
print("Correlation between polarity and SIN delta with 30 day lag:", SINlag3["own polarity"].corr(SINlag3["Delta"]))

c3 = SINlag3["own polarity"]
d3 = SINlag3["Delta"]

SINlag3ttest = ttest(c3, d3)

print(SINlag3ttest)

