"""

TIME SERIES ANALYSIS FILE

ALEXANDER LUND - THESIS ESCP

"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import plotly.offline
import quandl
import yfinance
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


# Time series analysis on sentiments

NLPdf = pd.read_csv("~/Desktop/processed-reddit-text-thesis.csv")

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

invescoESGdf["Delta"] = invescoESGdf["Open"] - invescoESGdf["Close"]

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

testdf = NLPdf.merge(ESGdf, on="datetime")

#print(testdf["own polarity"].corr(testdf["Delta"]))

testdf.plot.scatter(x="own polarity", y="standard_delta")
plt.show()