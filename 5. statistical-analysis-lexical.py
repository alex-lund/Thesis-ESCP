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
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import chi2_contingency
from bioinfokit.analys import stat

#### sentiment df preparation ####

#reddit
redditdf = pd.read_csv("4. processed-reddit-text-thesis.csv")

redditdf.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "cleaned body", "own subjectivity"], inplace=True)

redditdf["datetime"] = pd.to_datetime(redditdf["datetime"]).dt.date


### graphing of polarity
"""redditgraphdf = redditdf.groupby("datetime").mean()
redditgraphdf.reset_index(inplace=True)

redditgraph = px.line(redditgraphdf,x="datetime",y="own polarity")
redditgraphsign = px.line(redditgraphdf,x="datetime",y="polarity_sign")

redditgraph.show()
redditgraphsign.show()"""


# fund information and computation of delta
InvescoESG = yfinance.Ticker("ESG") #tried with ESG, SUSA, ICLN, ESGV, ESGD, PBD, WOOD, EVX, RNRG => all funds return same correlation pattern with sentiment-polarity

invescoESGdf = InvescoESG.history(start='2010-1-1', end='2021-1-1') #(start='2010-1-1', end='2021-1-1')

invescoESGdf = invescoESGdf.drop(columns=["Volume", "Dividends", "Stock Splits"])

invescoESGdf = invescoESGdf.reset_index()

for i in ['Open', 'High', 'Close', 'Low']:
      invescoESGdf[i]  =  invescoESGdf[i].astype('float64')

invescoESGdf["Delta"] = invescoESGdf["Close"] - invescoESGdf["Open"]


sin = yfinance.Ticker("PM") #tried with PM, ITA, EAFE, LMT, BP,  XOP, XLE, VDE

sin = sin.history(start='2010-1-1', end='2021-1-1')

sin = sin.drop(columns=["Volume", "Dividends", "Stock Splits"])

sin = sin.reset_index()

for i in ['Open', 'High', 'Close', 'Low']:
      sin[i]  =  sin[i].astype('float64')

sin["Delta"] = sin["Close"] - sin["Open"]



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
ESG = redditdf.merge(ESGdf, on="datetime")
SIN = redditdf.merge(SINdf, on="datetime")


# computation of mean polarity groupedby date, meaning we take average polarity for each day to avoid granularity
ESG = ESG.groupby("datetime").mean()
ESG.reset_index(inplace=True)

SIN = SIN.groupby("datetime").mean()
SIN.reset_index(inplace=True)

"""ESG.drop(columns=["polarity_sign"], inplace=True)
SIN.drop(columns=["polarity_sign"], inplace=True)"""


### MEANS, STDEVs & HISTOGRAMS ###

#for ESG
"""ESG_polarity_hist = ESG.hist(column="own polarity")
plt.show()

ESG_Delta_hist = ESG.hist(column="Delta")
plt.show()"""

#for SIN
"""
SIN_polarity_hist = SIN.hist(column="own polarity")
plt.show()

SIN_Delta_hist = SIN.hist(column="Delta")
plt.show()
"""
print(shapiro(ESG["own polarity"]))
print(shapiro(ESG["Delta"]))


### CATEGORICAL VARIABLE CREATION ###

##
filt_pos1 = ESG["own polarity"] > 0
filt_neg1 = ESG["own polarity"] < 0
filt_zero1 = ESG["own polarity"] == 0

filt_pos2 = SIN["own polarity"] > 0
filt_neg2 = SIN["own polarity"] < 0
filt_zero2 = SIN["own polarity"] == 0

ESG["sentiment-direction"] = "undefined"
SIN["sentiment-direction"] = "undefined"

ESG.loc[filt_pos1, "sentiment-direction"]= "Bullish"
ESG.loc[filt_neg1, "sentiment-direction"]= "Bearish"
ESG.loc[filt_zero1, "sentiment-direction"]= "Neutral"

SIN.loc[filt_pos2, "sentiment-direction"]= "Bullish"
SIN.loc[filt_neg2, "sentiment-direction"]= "Bearish"
SIN.loc[filt_zero2, "sentiment-direction"]= "Neutral"

##
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


#### ARIMA ####

ESGARIMA = ESG.drop(columns=["body word count", "polarity_sign", "sentiment-direction", "fund-direction"])
ESGARIMA.set_index("datetime",inplace=True)

adf_pola = adfuller(ESGARIMA["own polarity"])
adf_delta = adfuller(ESGARIMA["Delta"])

print(adf_pola)
print(adf_delta)

"""ESGARIMA.plot()
plt.show()"""



#### SPEARMAN CORRELATION ASSESSMENT (CONTINOUS-CONTINUOUS) #######
"""
# FOR ESG
ESG_corr = spearmanr(ESG["own polarity"],ESG["Delta"])

print("ESG-Sentiment correlation on intraday basis: ",ESG_corr)

ESGreactive = ESG.copy(deep=False)
ESGreactive[["reactive_lag_polarity"]] = ESGreactive["own polarity"].shift(-1)
ESGreactive = ESGreactive.dropna()

ESGreactive_corr = spearmanr(ESGreactive["reactive_lag_polarity"],ESGreactive["Delta"])

print("ESG-Sentiment correlation on 1 day reactive basis: ", ESGreactive_corr)

ESGproactive = ESG.copy(deep=False)
ESGproactive[["proactive_lag_polarity"]] = ESGproactive["own polarity"].shift(1)
ESGproactive = ESGproactive.dropna()

ESGproactive_corr = spearmanr(ESGproactive["proactive_lag_polarity"],ESGproactive["Delta"])


print("ESG-Sentiment correlation on 1 day proactive basis: ", ESGproactive_corr)

ESGrollingavg = ESG.copy(deep=False)
ESGrollingavg[["rolling_avg_polarity"]] = ESGrollingavg["own polarity"].rolling(5).mean()
ESGrollingavg = ESGrollingavg.dropna()

ESGrollingavg_corr = spearmanr(ESGrollingavg["rolling_avg_polarity"],ESGrollingavg["Delta"])

print("ESG-Sentiment correlation on 5 day rolling average lag basis: ", ESGrollingavg_corr)

print("\n")

# FOR SIN
SIN_corr = spearmanr(SIN["own polarity"],SIN["Delta"])

print("SIN-Sentiment correlation on intraday basis: ",SIN_corr)

SINreactive = SIN.copy(deep=False)
SINreactive[["reactive_lag_polarity"]] = SINreactive["own polarity"].shift(-1)
SINreactive = SINreactive.dropna()

SINreactive_corr = spearmanr(SINreactive["reactive_lag_polarity"],SINreactive["Delta"])

print("SIN-Sentiment correlation on reactive lag basis: ",SINreactive_corr)

SINproactive = SIN.copy(deep=False)
SINproactive[["proactive_lag_polarity"]] = SINproactive["own polarity"].shift(1)
SINproactive = SINproactive.dropna()

SINproactive_corr = spearmanr(SINproactive["proactive_lag_polarity"],SINproactive["Delta"])

print("SIN-Sentiment correlation on proactive lag basis: ", SINproactive_corr)

SINrollingavg = SIN.copy(deep=False)
SINrollingavg[["rolling_avg_polarity"]] = SINrollingavg["own polarity"].rolling(5).mean()
SINrollingavg = SINrollingavg.dropna()

SINrollingavg_corr = spearmanr(SINrollingavg["rolling_avg_polarity"],SINrollingavg["Delta"])

print("SIN-Sentiment correlation on 5 day rolling average lag basis: ", SINrollingavg_corr)

print("\n")
"""
#### LOGISTIC APPROACH (CONTINUOUS-CATEGORICAL) #######

### Multinomial Logistic Regression ###
"""
## FOR ESG


ESG_logit = ESG.copy(deep=False)
ESG_logit_reactive = ESG.copy(deep=False)
ESG_logit_proactive = ESG.copy(deep=False)

# non-lagged
ESG_logit = ESG_logit.drop(columns=["datetime", "body word count", "own polarity", "polarity_sign", "fund-direction"])
ESG_logit = pd.get_dummies(data=ESG_logit, prefix="", columns=["sentiment-direction"])

ESG_log_Y = ESG_logit[["_Neutral", "_Bullish", "_Bearish"]]
ESG_log_X = ESG_logit.drop(columns=[ "_Bearish", "_Bullish", "_Neutral"])

logit_model_esg =sm.MNLogit(ESG_log_Y,sm.add_constant(ESG_log_X))
result1=logit_model_esg.fit()
stats1=result1.summary()
print(stats1)
print(stats1.as_latex())

# reactive
ESG_logit_reactive = ESG_logit_reactive["sentiment-direction"].shift(-1)
ESG_logit_reactive = ESG_logit_reactive.drop(columns=["datetime", "body word count", "own polarity", "polarity_sign", "fund-direction"])
ESG_logit_reactive = pd.get_dummies(data=ESG_logit_reactive, prefix="", columns=["sentiment-direction"])

ESG_log_Y_reactive = ESG_logit_reactive[["_Neutral", "_Bullish", "_Bearish"]]
ESG_log_X_reactive = ESG_logit_reactive.drop(columns=[ "_Bearish", "_Bullish", "_Neutral"])

logit_model_esg_reac =sm.MNLogit(ESG_log_Y_reactive,sm.add_constant(ESG_log_X_reactive))
result2=logit_model_esg_reac.fit()
stats2=result2.summary()
print(stats2)
print(stats2.as_latex())

# proactive
ESG_logit_proactive = ESG_logit_proactive["sentiment-direction"].shift(1)
ESG_logit_proactive = ESG_logit_proactive.dropna()
ESG_logit_proactive = ESG_logit_proactive.drop(columns=["datetime", "body word count", "own polarity", "polarity_sign", "fund-direction"])
ESG_logit_proactive = pd.get_dummies(data=ESG_logit_proactive, prefix="", columns=["sentiment-direction"])

ESG_log_Y_proactive = ESG_logit_proactive[["_Neutral", "_Bullish", "_Bearish"]]
ESG_log_X_proactive = ESG_logit_proactive.drop(columns=[ "_Bearish", "_Bullish", "_Neutral"])

logit_model_esg_proac =sm.MNLogit(ESG_log_Y_proactive,sm.add_constant(ESG_log_X_proactive))
result3=logit_model_esg_proac.fit()
stats3=result3.summary()
print(stats3)
print(stats3.as_latex())

## FOR SIN

SIN_logit = SIN.copy(deep=False)
SIN_logit_reactive = SIN.copy(deep=False)
SIN_logit_proactive = SIN.copy(deep=False)

# non-lagged
SIN_logit = SIN_logit.drop(columns=["datetime", "body word count", "own polarity", "polarity_sign", "fund-direction"])
SIN_logit = pd.get_dummies(data=SIN_logit, prefix="", columns=["sentiment-direction"])

SIN_log_Y = SIN_logit[["_Neutral", "_Bullish", "_Bearish"]]
SIN_log_X = SIN_logit.drop(columns=[ "_Bearish", "_Bullish", "_Neutral"])

logit_model_sin = sm.MNLogit(SIN_log_Y,sm.add_constant(SIN_log_X))
result4 = logit_model_sin.fit()
stats4 = result4.summary()
print(stats4)
print(stats4.as_latex())

# reactive
SIN_logit_reactive = SIN_logit_reactive["sentiment-direction"].shift(-1)
SIN_logit_reactive = SIN_logit_reactive.drop(columns=["datetime", "body word count", "own polarity", "polarity_sign", "fund-direction"])
SIN_logit_reactive = pd.get_dummies(data=SIN_logit_reactive, prefix="", columns=["sentiment-direction"])

SIN_log_Y_reactive = SIN_logit_reactive[["_Neutral", "_Bullish", "_Bearish"]]
SIN_log_X_reactive = SIN_logit_reactive.drop(columns=[ "_Bearish", "_Bullish", "_Neutral"])

logit_model_sin_reac =sm.MNLogit(SIN_log_Y_reactive,sm.add_constant(SIN_log_X_reactive))
result5=logit_model_sin_reac.fit()
stats5=result5.summary()
print(stats5)
print(stats5.as_latex())

# proactive
SIN_logit_proactive = SIN_logit_proactive["sentiment-direction"].shift(1)
SIN_logit_proactive.dropna()
SIN_logit_proactive = SIN_logit_proactive.drop(columns=["datetime", "body word count", "own polarity", "polarity_sign", "fund-direction"])
SIN_logit_proactive = pd.get_dummies(data=SIN_logit_proactive, prefix="", columns=["sentiment-direction"])

SIN_log_Y_proactive = SIN_logit_proactive[["_Neutral", "_Bullish", "_Bearish"]]
SIN_log_X_proactive = SIN_logit_proactive.drop(columns=[ "_Bearish", "_Bullish", "_Neutral"])

logit_model_sin_proac = sm.MNLogit(SIN_log_Y_proactive,sm.add_constant(SIN_log_X_proactive))
result6 = logit_model_sin_proac.fit()
stats6 = result6.summary()
print(stats6)
print(stats6.as_latex())
"""
### Binomial Logistic Regression ###

# FOR ESG
"""
ESG_logit = pd.get_dummies(data=ESG,prefix="", columns=["sentiment-direction"])

ESGbearishlogit = sm.Logit(ESG_logit["_Bearish"], ESG_logit["Delta"]).fit()

ESGbullishlogit = sm.Logit(ESG_logit["_Bullish"], ESG_logit["Delta"]).fit()

ESGneutrallogit = sm.Logit(ESG_logit["_Neutral"], ESG_logit["Delta"]).fit()

print(ESGbearishlogit.summary())
print(ESGbullishlogit.summary())
print(ESGneutrallogit.summary())

ESG_logit_reactive = ESG_logit.copy(deep=False)
ESG_logit_reactive["_Bearish"] = ESG_logit_reactive["_Bearish"].shift(-1)
ESG_logit_reactive["_Bullish"] = ESG_logit_reactive["_Bullish"].shift(-1)
ESG_logit_reactive["_Neutral"] = ESG_logit_reactive["_Neutral"].shift(-1)
ESG_logit_reactive.dropna(inplace=True)

ESGbearishlogitreactive = sm.Logit(ESG_logit_reactive["_Bearish"], ESG_logit_reactive["Delta"]).fit()

ESGbullishlogitreactive = sm.Logit(ESG_logit_reactive["_Bullish"], ESG_logit_reactive["Delta"]).fit()

ESGneutrallogitreactive = sm.Logit(ESG_logit_reactive["_Neutral"], ESG_logit_reactive["Delta"]).fit()

print(ESGbearishlogitreactive.summary())
print(ESGbullishlogitreactive.summary())
print(ESGneutrallogitreactive.summary())

ESG_logit_proactive = ESG_logit.copy(deep=False)
ESG_logit_proactive["_Bearish"] = ESG_logit_proactive["_Bearish"].shift(1)
ESG_logit_proactive["_Bullish"] = ESG_logit_proactive["_Bullish"].shift(1)
ESG_logit_proactive["_Neutral"] = ESG_logit_proactive["_Neutral"].shift(1)
ESG_logit_proactive.dropna(inplace=True)

ESGbearishlogitproactive = sm.Logit(ESG_logit_proactive["_Bearish"], ESG_logit_proactive["Delta"]).fit()

ESGbullishlogitproactive = sm.Logit(ESG_logit_proactive["_Bullish"], ESG_logit_proactive["Delta"]).fit()

ESGneutrallogitproactive = sm.Logit(ESG_logit_proactive["_Neutral"], ESG_logit_proactive["Delta"]).fit()

print(ESGbearishlogitproactive.summary())
print(ESGbullishlogitproactive.summary())
print(ESGneutrallogitproactive.summary())
"""

# FOR SIN
"""
SIN_logit = pd.get_dummies(data=SIN, prefix="", columns=["sentiment-direction"])

SINbearishlogit = sm.Logit(SIN_logit["_Bearish"], SIN_logit["Delta"]).fit()

SINbullishlogit = sm.Logit(SIN_logit["_Bullish"], SIN_logit["Delta"]).fit()

SINneutrallogit = sm.Logit(SIN_logit["_Neutral"], SIN_logit["Delta"]).fit()

print(SINbearishlogit.summary())
print(SINbullishlogit.summary())
print(SINneutrallogit.summary())

SIN_logit_reactive = SIN_logit.copy(deep=False)
SIN_logit_reactive["_Bearish"] = SIN_logit_reactive["_Bearish"].shift(-1)
SIN_logit_reactive["_Bullish"] = SIN_logit_reactive["_Bullish"].shift(-1)
SIN_logit_reactive["_Neutral"] = SIN_logit_reactive["_Neutral"].shift(-1)
SIN_logit_reactive.dropna(inplace=True)

SINbearishlogitreactive = sm.Logit(SIN_logit_reactive["_Bearish"], SIN_logit_reactive["Delta"]).fit()

SINbullishlogitreactive = sm.Logit(SIN_logit_reactive["_Bullish"], SIN_logit_reactive["Delta"]).fit()

SINneutrallogitreactive = sm.Logit(SIN_logit_reactive["_Neutral"], SIN_logit_reactive["Delta"]).fit()

print(SINbearishlogitreactive.summary())
print(SINbullishlogitreactive.summary())
print(SINneutrallogitreactive.summary())

SIN_logit_proactive = SIN_logit.copy(deep=False)
SIN_logit_proactive["_Bearish"] = SIN_logit_proactive["_Bearish"].shift(1)
SIN_logit_proactive["_Bullish"] = SIN_logit_proactive["_Bullish"].shift(1)
SIN_logit_proactive["_Neutral"] = SIN_logit_proactive["_Neutral"].shift(1)
SIN_logit_proactive.dropna(inplace=True)

SINbearishlogitproactive = sm.Logit(SIN_logit_proactive["_Bearish"], SIN_logit_proactive["Delta"]).fit()

SINbullishlogitproactive = sm.Logit(SIN_logit_proactive["_Bullish"], SIN_logit_proactive["Delta"]).fit()

SINneutrallogitproactive = sm.Logit(SIN_logit_proactive["_Neutral"], SIN_logit_proactive["Delta"]).fit()

print(SINbearishlogitproactive.summary())
print(SINbullishlogitproactive.summary())
print(SINneutrallogitproactive.summary())
"""
### LATEX Rendering ####
"""
#
print("ESG Logit Intraday Bearish Latex Results: ")
print(ESGbearishlogit.summary().as_latex())
print("\n")

print("SIN Logit Intraday Bearish Latex Results: ")
print(SINbearishlogit.summary().as_latex())
print("\n")

#
print("ESG Logit Intraday Bullish Latex Results: ")
print(ESGbullishlogit.summary().as_latex())
print("\n")

print("SIN Logit Intraday Bullish Latex Results: ")
print(SINbullishlogit.summary().as_latex())
print("\n")

#
print("ESG Logit Intraday Neutral Latex Results: ")
print(ESGneutrallogit.summary().as_latex())
print("\n")

print("SIN Logit Intraday Neutral Latex Results: ")
print(SINneutrallogit.summary().as_latex())
print("\n")

#
print("ESG Logit Reactive Bearish Latex Results: ")
print(ESGbearishlogitreactive.summary().as_latex())
print("\n")

print("SIN Logit Reactive Bearish Latex Results: ")
print(SINbearishlogitreactive.summary().as_latex())
print("\n")

#
print("ESG Logit Reactive Bullish Latex Results: ")
print(ESGbullishlogitreactive.summary().as_latex())
print("\n")

print("SIN Logit Reactive Bullish Latex Results: ")
print(SINbullishlogitreactive.summary().as_latex())
print("\n")

#
print("ESG Logit Reactive Neutral Latex Results: ")
print(ESGneutrallogitreactive.summary().as_latex())
print("\n")

print("SIN Logit Reactive Neutral Latex Results: ")
print(SINneutrallogitreactive.summary().as_latex())
print("\n")

#
print("ESG Logit Proactive Bearish Latex Results: ")
print(ESGbearishlogitproactive.summary().as_latex())
print("\n")

print("SIN Logit Proactive Bearish Latex Results: ")
print(SINbearishlogitproactive.summary().as_latex())
print("\n")

#
print("ESG Logit Proactive Bullish Latex Results: ")
print(ESGbullishlogitproactive.summary().as_latex())
print("\n")

print("SIN Logit Proactive Bullish Latex Results: ")
print(SINbullishlogitproactive.summary().as_latex())
print("\n")

#
print("ESG Logit Proactive Neutral Latex Results: ")
print(ESGneutrallogitproactive.summary().as_latex())
print("\n")

print("SIN Logit Proactive Neutral Latex Results: ")
print(SINneutrallogitproactive.summary().as_latex())
print("\n")"""


#### LOGISTIC APPROACH (CATEGORICAL-CATEGORICAL) #######

# FOR ESG
ESG_chi = ESG.drop(columns=["datetime","own polarity", "Delta"])

ESG_contingency = pd.crosstab(index=ESG_chi["sentiment-direction"], columns=ESG_chi["fund-direction"])
print(ESG_contingency)

ESG_observed = np.array([[17, 80, 79], [18, 83, 88], [7, 18, 13]])
ESG_chi_val, ESG_p_val, ESG_dof, ESG_expected = chi2_contingency(ESG_observed)
print(ESG_chi_val, ESG_p_val, ESG_dof, ESG_expected)

# FOR SIN
SIN_chi = SIN.drop(columns=["datetime","own polarity", "Delta"])

SIN_contingency = pd.crosstab(index=SIN_chi["sentiment-direction"], columns=SIN_chi["fund-direction"])
print(SIN_contingency)

SIN_observed = np.array([[1, 88, 78], [2, 75, 85], [0, 18, 11]])
SIN_chi_val, SIN_p_val, SIN_dof, SIN_expected = chi2_contingency(SIN_observed)
print(SIN_chi_val, SIN_p_val, SIN_dof, SIN_expected)

print(ESG_contingency.to_latex())
print(SIN_contingency.to_latex())
