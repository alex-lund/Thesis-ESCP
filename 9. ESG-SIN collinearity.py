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


# fund information and computation of delta
InvescoESG = yfinance.Ticker("SUSA") #tried with ESG, SUSA, ICLN, ESGV, ESGD, PBD, WOOD, EVX, RNRG => all funds return same correlation pattern with sentiment-polarity

invescoESGdf = InvescoESG.history(start='2010-1-1', end='2021-1-1') #(start='2010-1-1', end='2021-1-1')

invescoESGdf = invescoESGdf.drop(columns=["Volume", "Dividends", "Stock Splits"])

invescoESGdf = invescoESGdf.reset_index()

for i in ['Open', 'High', 'Close', 'Low']:
      invescoESGdf[i]  =  invescoESGdf[i].astype('float64')

invescoESGdf["ESG_delta"] = invescoESGdf["Close"] - invescoESGdf["Open"]


sin = yfinance.Ticker("VICE")

sin = sin.history(start='2010-1-1', end='2021-1-1')

sin = sin.drop(columns=["Volume", "Dividends", "Stock Splits"])

sin = sin.reset_index()

for i in ['Open', 'High', 'Close', 'Low']:
      sin[i]  =  sin[i].astype('float64')

sin["SIN_delta"] = sin["Close"] - sin["Open"]

ESGdf = invescoESGdf.drop(columns=["Open", "High", "Low", "Close"])

ESGdf["Date"] = pd.to_datetime(ESGdf["Date"]).dt.date
ESGdf.rename(columns={"Date": "datetime"}, inplace=True)

SINdf = sin.drop(columns=["Open", "High", "Low", "Close"])

SINdf["Date"] = pd.to_datetime(SINdf["Date"]).dt.date
SINdf.rename(columns={"Date": "datetime"}, inplace=True)

esg_sin_df = ESGdf.merge(SINdf, on="datetime")

esg_sin_df.plot()
plt.show()


# computation of mean polarity groupedby date, meaning we take average polarity for each day to avoid granularity


filt_pos3 = esg_sin_df["ESG_delta"] > 0
filt_neg3 = esg_sin_df["ESG_delta"] < 0
filt_zero3 = esg_sin_df["ESG_delta"] == 0
filt_pos4 = esg_sin_df["SIN_delta"] > 0
filt_neg4 = esg_sin_df["SIN_delta"] < 0
filt_zero4 = esg_sin_df["SIN_delta"] == 0

esg_sin_df["ESG-fund-direction"] = "undefined"
esg_sin_df["SIN-fund-direction"] = "undefined"

esg_sin_df.loc[filt_pos3, "ESG-fund-direction"]= "Increase"
esg_sin_df.loc[filt_neg3, "ESG-fund-direction"]= "Decrease"
esg_sin_df.loc[filt_zero3, "ESG-fund-direction"]= "Constant"

esg_sin_df.loc[filt_pos4, "SIN-fund-direction"]= "Increase"
esg_sin_df.loc[filt_neg4, "SIN-fund-direction"]= "Decrease"
esg_sin_df.loc[filt_zero4, "SIN-fund-direction"]= "Constant"

esg_sin_df = esg_sin_df[esg_sin_df["ESG-fund-direction"] != "Constant"]
esg_sin_df = esg_sin_df[esg_sin_df["SIN-fund-direction"] != "Constant"]


contingency = pd.crosstab(index=esg_sin_df["ESG-fund-direction"], columns=esg_sin_df["SIN-fund-direction"])
print(contingency)
print(contingency.to_latex())

observed = [contingency.to_numpy()[0], contingency.to_numpy()[1]]
chi_val, p_val, dof, expected = chi2_contingency(observed)
print(chi_val, p_val, dof, expected )




