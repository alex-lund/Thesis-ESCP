"""

TIME SERIES ANALYSIS FILE

ALEXANDER LUND - THESIS ESCP

"""
import numpy as np
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
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Time series analysis on sentiments

NLPdf = pd.read_csv("processed-reddit-text-thesis.csv")

NLPdf.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "cleaned body", "own subjectivity"], inplace=True)

NLPdf["datetime"] = pd.to_datetime(NLPdf["datetime"]).dt.date

#computation of mean already here?
NLPdfnew = NLPdf.groupby("datetime").mean()

NLPdfnew.reset_index(inplace=True)

print(NLPdfnew["own polarity"].value_counts())

graph = px.line(NLPdfnew,x="datetime",y="own polarity")
graphsign = px.line(NLPdfnew,x="datetime",y="polarity_sign")

#graph.show()
#graphsign.show()

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

# removal of unnecessary info from both ESG & SIN etfs
ESGdf = invescoESGdf.drop(columns=["Open", "High", "Low", "Close"])

ESGdf["Date"] = pd.to_datetime(ESGdf["Date"]).dt.date
ESGdf.rename(columns={"Date": "datetime"}, inplace=True)

SINdf = sin.drop(columns=["Open", "High", "Low", "Close"])

SINdf["Date"] = pd.to_datetime(SINdf["Date"]).dt.date
SINdf.rename(columns={"Date": "datetime"}, inplace=True)

# merger / creation of ESG dataset (combination of NLP polarity data with etf delta)
ESG = NLPdf.merge(ESGdf, on="datetime")
SIN = NLPdf.merge(SINdf, on="datetime")

# what is important to note here, is that we take the average polarity
ESG = ESG.groupby("datetime").mean()
ESG.reset_index(inplace=True)

SIN = SIN.groupby("datetime").mean()
SIN.reset_index(inplace=True)

### MEANS, STDEVs & HISTOGRAMS ###

#for ESG
"""ESG_polarity_hist = ESG.hist(column="own polarity")
plt.show()
ESG_Delta_hist = ESG.hist(column="Delta")
plt.show()"""

#for SIN
"""SIN_polarity_hist = SIN.hist(column="own polarity")
plt.show()
SIN_Delta_hist = SIN.hist(column="Delta")
plt.show()"""



#ESG = ESG[(ESG['datetime'] < '2020-02-01') & (ESG['datetime'] < '2020-06-01')]

ESGcorr = ESG["own polarity"].corr(ESG["Delta"])
print("Correlation between polarity and ESG delta on intraday basis:",ESGcorr)
ESGcorr2 = ESG["polarity_sign"].corr(ESG["Delta"])
print("Correlation with Bullish/Bearish apprehension: ", ESGcorr2)

a = ESG["own polarity"]
b = ESG["Delta"]

ESGttest = ttest(a, b)

print(ESGttest)

lm = ols("Q('own polarity') ~ Delta", data=ESG).fit()
table = sm.stats.anova_lm(lm)

print(table)


SINcorr = SIN["own polarity"].corr(SIN["Delta"])
print("Correlation between polarity and SIN delta on intraday basis:", SINcorr)

c = SIN["own polarity"]
d = SIN["Delta"]

SINttest = ttest(c, d)

print(SINttest)
print("\n")

ESGlag1 = ESG
ESGlag1[["Delta", "standard_delta", "minmax_std_delta"]] = ESG[["Delta", "standard_delta", "minmax_std_delta"]].shift(1)
ESGlag1 = ESGlag1.dropna()
ESGlag1corr = ESGlag1["own polarity"].corr(ESGlag1["Delta"])
print("Correlation between polarity and ESG delta with 1 day lag:", ESGlag1corr)
ESGbulbearlag1corr = ESGlag1["polarity_sign"].corr(ESGlag1["Delta"])
print("Correlation with Bullish/Bearish apprehension + lag factor 1: ", ESGbulbearlag1corr)
a1 = ESGlag1["own polarity"]
b1 = ESGlag1["Delta"]

ESGlag1ttest = ttest(a1, b1)

print(ESGlag1ttest)

lmlag = ols("Q('polarity_sign') ~ Delta", data=ESGlag1).fit()
table2 = sm.stats.anova_lm(lm)

print(table2)


SINlag1 = SIN
SINlag1[["Delta"]] = SINlag1[["Delta"]].shift(1)
SINlag1 = SINlag1.dropna()
SINlag1corr = SINlag1["own polarity"].corr(SINlag1["Delta"])
print("Correlation between polarity and SIN delta with 1 day lag:", SINlag1corr)

c1 = SINlag1["own polarity"]
d1 = SINlag1["Delta"]

SINlag1ttest = ttest(c1, d1)

print(SINlag1ttest)
print("\n")



#ESGfinaldf = pd.DataFrame(columns=)

#testresults = pd.DataFrame
