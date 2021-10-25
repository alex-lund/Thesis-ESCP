"""

IMPACT ANALYSIS OF ESG FUND VS REAL TIME EMISSIONS

ALEXANDER LUND - THESIS ESCP

"""
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
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



# fund information
InvescoESG = yfinance.Ticker("ESG")

esg = InvescoESG.history(start='2010-1-1', end='2021-1-1')

esg = esg.drop(columns=["Volume", "Dividends", "Stock Splits"])

esg = esg.reset_index()

for i in ['Open', 'High', 'Close', 'Low']:
      esg[i]  =  esg[i].astype('float64')

esg["Delta"] = esg["Close"] - esg["Open"]


sin = yfinance.Ticker("VICE")

sin = sin.history(start='2010-1-1', end='2021-1-1')

sin = sin.drop(columns=["Volume", "Dividends", "Stock Splits"])

sin = sin.reset_index()

for i in ['Open', 'High', 'Close', 'Low']:
      sin[i]  =  sin[i].astype('float64')

sin["Delta"] = sin["Close"] - sin["Open"]

# emissions dataframe, data provided by OECD stats

emissions = pd.read_csv("AIR_GHG_11102021222842551.csv")

co2 = emissions[emissions["POL"] == "CO2"]

co2.drop(columns=["COU", "Country", "POL", "Pollutant", "VAR", "Variable", "YEA", "Unit Code", "PowerCode Code", "PowerCode", "Reference Period Code", "Reference Period", "Flag Codes", "Flags"], inplace=True)

ghg = emissions.loc[emissions["POL"] == "GHG"]

ghg.drop(columns=["COU", "Country", "POL", "Pollutant", "VAR", "Variable", "YEA", "Unit Code", "PowerCode Code", "PowerCode", "Reference Period Code", "Reference Period", "Flag Codes", "Flags"], inplace=True)
