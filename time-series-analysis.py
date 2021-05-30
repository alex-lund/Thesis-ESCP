"""

TIME SERIES ANALYSIS FILE (INDEX)

ALEXANDER LUND - THESIS ESCP

"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import quandl
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

NLPdf = pd.read_csv("~/Desktop/processed-text-thesis.csv")

#quandl.ApiConfig.api_key = "***"

InvescoESG = yf.Ticker("ESG")

invescoESGdf = InvescoESG.history(period='1d', start='2010-1-1', end='2020-10-1')

invescoESGdf = invescoESGdf.drop(columns=["Volume", "Dividends", "Stock Splits"])

invescoESGdf = invescoESGdf.reset_index()

for i in ['Open', 'High', 'Close', 'Low']:
      invescoESGdf[i]  =  invescoESGdf[i].astype('float64')

fig = go.Figure(data=[go.Candlestick(x=invescoESGdf["Date"],
                                     open= invescoESGdf["Open"],
                                     high= invescoESGdf["High"],
                                     low= invescoESGdf["Low"],
                                     close= invescoESGdf["Close"])])

#fig.show()

NLPdf["timestamp"] = NLPdf["timestamp"].astype("float64")

NLPdf["timestamp"] = NLPdf["timestamp"].astype("int64")

for timestamp in NLPdf["timestamp"]:
    timestamp.strip(".0")

#NLPdf['timestamp'] = NLPdf['timestamp'].apply(lambda x: pd.Timestamp(x).strftime('%Y-%m'))

#NLPdf.loc[:, "timestamp"] = pd.to_datetime(NLPdf["timestamp"])

