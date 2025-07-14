import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

"""

This file computes various technical indicators and uses them to generate signals for opening and closing leveraged long positions on Bitcoin. 

The file simulates leveraged trades and tracks the growth of the portfolio over a set period of historial data.

"""

def compute_macd(prices, short=12, long=26, signal=9):

    """Calcualtes the Moving Avergae COnvergence/Divergence (MACD)"""
    ema_short = prices.ewm(span=short, adjust=False).mean()
    ema_long = prices.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal


def compute_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_adx(df, period=14):
    """Calculates the Average Directional Index (ADX) for the data"""
    
    high = df['High', 'BTC-USD']
    low = df['Low', 'BTC-USD']
    close = df['Close', 'BTC-USD']

    #Calculate True Range (TR)
    tr1 = high.values - low.values
    tr2 = np.abs(high.values - close.shift(1).values)
    tr3 = np.abs(low.values - close.shift(1).values)
    df['TR'] = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3)), index=df.index)

    #Calculate Directional Movement (+DM and -DM)
    plus_dm = (high.diff()).where(high.diff() > low.diff().abs(), 0).where(high.diff() > 0, 0)
    minus_dm = (low.diff()).where(low.diff() < high.diff(), 0).where(low.diff() < 0, 0)
    
    df['+DM'] = plus_dm
    df['-DM'] = minus_dm.abs()

    #Calculate EMA for +DM, -DM, and TR
    df['TR_EMA'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    df['+DM_EMA'] = df['+DM'].ewm(alpha=1/period, adjust=False).mean()
    df['-DM_EMA'] = df['-DM'].ewm(alpha=1/period, adjust=False).mean()

    #Calculate DI+ and DI-
    df['+DI'] = (df['+DM_EMA'] / df['TR_EMA']) * 100
    df['-DI'] = (df['-DM_EMA'] / df['TR_EMA']) * 100

    #Calculate DX (Fixed line: using +DI and -DI)
    #Ensure no division by zero if DI+ and DI- sum to 0
    df['DI_Sum'] = df['+DI'] + df['-DI']
    df['DX'] = (np.abs(df['+DI'] - df['-DI']) / df['DI_Sum'].replace(0, 1)) * 100 # Handle division by zero 

    #Calculate ADX (EMA of DX)
    df['ADX'] = df['DX'].ewm(alpha=1/period, adjust=False).mean()
    return df['ADX']


ticker = "BTC-USD"
#Download a period of data with specific intervals
df = yf.download(ticker, interval="1d", period="2y")  

print(df.head())
print(df.columns)
print(type(df['Close', 'BTC-USD']))



df['ema10'] = df['Close', 'BTC-USD'].ewm(span=10, adjust=False).mean()
df['macd'], df['macd_signal'] = compute_macd(df['Close', 'BTC-USD'])
df['macd_hist'] = df['macd'] - df['macd_signal']
df['rsi'] = compute_rsi(df['Close', 'BTC-USD']) 
df['volume_ma20'] = df['Volume', 'BTC-USD'].rolling(window=20).mean() 
df['adx'] = compute_adx(df) 





df['signal'] = 0
in_position = False
phi = 1.618
df['ema21'] = df['Close', 'BTC-USD'].ewm(span=21, adjust=False).mean()

#golden ratio bands
df['golden_band_1'] = df['ema21'] * phi
df['golden_band_2'] = df['ema21'] * (phi ** 2)

leverage = 8  
liquidated = False
in_position = False
entry_price = None
liq_price = None
portfolio_value = 1000  

signals = []
pnl_history = []
for i in range(1, len(df)):
    

    # indicator confirmations
    macd_buy_signal = (df['macd'].iloc[i-1] < df['macd_signal'].iloc[i-1]) and \
                      (df['macd'].iloc[i] > df['macd_signal'].iloc[i])

    
    rsi_confirmation = (df['rsi'].iloc[i] > 50)

    
    volume_confirmation = (df['Volume', 'BTC-USD'].iloc[i] > df['volume_ma20'].iloc[i])

    adx_confirmation = (df['adx'].iloc[i] > 25) 
    print(df['adx'].iloc[i])
    
    # trading decisions/liquidation
    if not in_position and macd_buy_signal and rsi_confirmation and volume_confirmation:
        in_position = True
        entry_price = df['Close', 'BTC-USD'].iloc[i]
        liq_price = entry_price * (1 - 1/leverage)
        print(f"Entry at {float(entry_price):.2f}, Liquidation at {float(liq_price):.2f} | Confirmed by RSI({df['rsi'].iloc[i]:.2f}) and Volume")
        signals.append(1)
    
    elif in_position and df['Low', 'BTC-USD'].iloc[i] <= liq_price:


        print(f"Liquidated at {df['Close', 'BTC-USD'].iloc[i]:.2f}")
        portfolio_value = 0
        liquidated = True
        signals.append(0)


    elif in_position and (df['Close', 'BTC-USD'].iloc[i] < df['ema10'].iloc[i]):

        pnl = leverage*(df['Close', 'BTC-USD'].iloc[i] - entry_price)/entry_price
        print('pnl', pnl)
        pnl_history.append(pnl)
        
        portfolio_value *= (1+pnl)
        print(portfolio_value)
        print('SELL')
        in_position = False
        signals.append(0)
    elif in_position and (df['Close', 'BTC-USD'].iloc[i] >= df['ema10'].iloc[i]):
        print(df['Close', 'BTC-USD'].iloc[i])
    else:
        
        df.at[df.index[i], 'signal'] = np.nan  
        signals.append(np.nan)


df = df.iloc[1:len(signals)+1] 
df['signal'] = signals
print("portfolio value", portfolio_value)
print(pnl_history)
cumulative = []
for i in range(len(pnl_history)):

    cumulative.append(sum(pnl_history[:i+1]))
print(cumulative)

