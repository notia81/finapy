import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas_datareader import data
data = data.DataReader(['XOM', 'CVX'], 'yahoo', '01-01-2010', '12-31-2015')
data['normP'] = data['Adj Close']/data['Adj Close'].shift(1)
data['normP']['XOM'][0] = 1
data['normP']['CVX'][0] = 1
data['normP'] = np.cumproduct(data['normP'])

norm_spread = data['normP']['XOM']-data['normP']['CVX']
spread = data['Adj Close']['XOM']-data['Adj Close']['CVX']

window = 150

norm_spread = (norm_spread - norm_spread.rolling(window = window , min_periods = window,
                center =False).mean())/norm_spread.rolling(window = window , min_periods = window , center = False).std()
spread = (spread - spread.rolling(window = window, min_periods = window, center = False).mean())/ \
    spread.rolling(window = window, min_periods = window, center = False).std()

# both are the same essentially.
"""
Need to see what is the convergence rate at different points in time
using these can generate short term trading signals
"""
# first see if they are I(1) overall?
print(sm.tsa.adfuller(data['Adj Close']['XOM']))
print(sm.tsa.adfuller(data['Adj Close']['CVX']))
# clearly I(1).
# Store at points in time when the series are cointegrated.

coint_fn = lambda x , y : sm.tsa.coint(x, y)

def apply_rolling_coint(x , y , fn = lambda a, b :sm.tsa.coint(a,b ), window=150):
    coint_sig = np.zeros(len(x.index[window:]))
    coint_coef = np.zeros(len(y.index[window:]))

    for i in range(window,len(x.index)):
        coint_coef[i-window] , coint_sig[i-window] , _ = fn(x.iloc[i-window:i] ,
                                                    y.iloc[i-window:i])
    coint_coef = pd.Series(coint_coef , index=x.index[window:])
    coint_sig = pd.Series(coint_sig , index=y.index[window:])
    return(coint_coef , coint_sig)

coint_coef , coint_sig = apply_rolling_coint(data['Adj Close']['XOM'], data['Adj Close']['CVX'])
coint_markers = coint_coef[coint_sig<=0.05]

def plot_coints(coint_coef, coint_markers, spread, window=150):
    # these plots show that when the coef is sig, then X can Predict Y
    # which can be used to close the gap
    fig , ax1 = plt.subplots()
    ax1.plot(coint_coef.index, coint_coef, 'g',alpha=0.6 )
    ax1.plot(coint_markers.index, coint_markers, 'o', alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(spread.index[window:] , spread.iloc[window:], color='orange', alpha=0.6)
    ax2.plot(spread.index[window:] , np.linspace(0,0,len(spread.iloc[window:])),color='red',alpha=0.6)
    plt.show()

