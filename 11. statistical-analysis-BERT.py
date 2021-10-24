

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
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import chi2_contingency
from bioinfokit.analys import stat

df = pd.read_csv("bert-predictions-labels-reddit.csv")

df.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.1.1", "body", "body word count", "own polarity", "own subjectivity", "polarity_sign"], inplace=True)
df.dropna(inplace=True)

df["datetime"] = pd.to_datetime(df["datetime"]).dt.date

print(df["BERT-label"].value_counts())

# fund information and computation of delta
InvescoESG = yfinance.Ticker("ESG") #tried with ESG, SUSA, ICLN, ESGV, ESGD, PBD, WOOD, EVX, RNRG => all funds return same correlation pattern with sentiment-polarity

invescoESGdf = InvescoESG.history(start='2010-1-1', end='2021-1-1') #(start='2010-1-1', end='2021-1-1')

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

ESGdf = invescoESGdf.drop(columns=["Open", "High", "Low", "Close"])

ESGdf["Date"] = pd.to_datetime(ESGdf["Date"]).dt.date
ESGdf.rename(columns={"Date": "datetime"}, inplace=True)

SINdf = sin.drop(columns=["Open", "High", "Low", "Close"])

SINdf["Date"] = pd.to_datetime(SINdf["Date"]).dt.date
SINdf.rename(columns={"Date": "datetime"}, inplace=True)

ESG = df.merge(ESGdf, on="datetime")
SIN = df.merge(SINdf, on="datetime")


# computation of mean polarity groupedby date, meaning we take average polarity for each day to avoid granularity


filt_pos3 = ESG["Delta"] > 0
filt_neg3 = ESG["Delta"] < 0
filt_zero3 = ESG["Delta"] == 0

filt_pos4 = SIN["Delta"] > 0
filt_neg4 = SIN["Delta"] < 0
filt_zero4 = SIN["Delta"] == 0

ESG["fund-direction"] = "undefined"
SIN["fund-direction"] = "undefined"

ESG.loc[filt_pos3, "fund-direction"]= "Increase"
ESG.loc[filt_neg3, "fund-direction"]= "Decrease"
ESG.loc[filt_zero3, "fund-direction"]= "Constant"

SIN.loc[filt_pos4, "fund-direction"]= "Increase"
SIN.loc[filt_neg4, "fund-direction"]= "Decrease"
SIN.loc[filt_zero4, "fund-direction"]= "Constant"

ESG = ESG[ESG["fund-direction"] != "Constant"]

SIN = SIN[SIN["fund-direction"] != "Constant"]

ESG_contingency = pd.crosstab(index=ESG["BERT-label"], columns=ESG["fund-direction"])
print(ESG_contingency)
print("\n")

print(ESG_contingency.to_latex())

ESG_observed = [ESG_contingency.to_numpy()[0], ESG_contingency.to_numpy()[1]]
ESG_chi_val, ESG_p_val, ESG_dof, ESG_expected = chi2_contingency(ESG_observed)
print(ESG_chi_val, ESG_p_val, ESG_dof, ESG_expected)

SIN_contingency = pd.crosstab(index=SIN["BERT-label"], columns=SIN["fund-direction"])
print(SIN_contingency)
print("\n")

print(SIN_contingency.to_latex())

SIN_observed = [SIN_contingency.to_numpy()[0], SIN_contingency.to_numpy()[1]]
SIN_chi_val, SIN_p_val, SIN_dof, SIN_expected = chi2_contingency(SIN_observed)
print(SIN_chi_val, SIN_p_val, SIN_dof, SIN_expected)

