import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
    Here is a portfolion simulator which measures the sharpe ratio of 10,000 randomly weighted portfolios over the given period.
    This allows us to find the portfolio weighting with the highest returns relative to volatility.

    

   
"""

tickers = [
    'AAPL',     # Apple
    'MSFT',     # Microsoft
    'GOOGL',    # Alphabet (Class A)
    'GOOG',     # Alphabet (Class C)
    'AMZN',     # Amazon
    'NVDA',     # NVIDIA
    'BRK-B',    # Berkshire Hathaway
    'TSLA',     # Tesla
    'META',     # Meta Platforms
    'LLY',      # Eli Lilly
    'JPM',      # JPMorgan Chase
    'V',        # Visa
    'JNJ',      # Johnson & Johnson
    'UNH',      # UnitedHealth Group
    'XOM',      # ExxonMobil
    'WMT',      # Walmart
    'MA',       # Mastercard
    'PG',       # Procter & Gamble
    'AVGO',     # Broadcom
    'HD',       # Home Depot
    'MRK',      # Merck
    'ORCL',     # Oracle
    'CVX',      # Chevron
    'PEP',      # PepsiCo
    'ABBV',     # AbbVie
    'KO',       # Coca-Cola
    'COST',     # Costco
    'BAC',      # Bank of America
    'ASML',     # ASML (Netherlands)
    'TM',       # Toyota (Japan)
    'TMO',      # Thermo Fisher
    'ADBE',     # Adobe
    'CSCO',     # Cisco
    'NVO',      # Novo Nordisk (Denmark)
    'NFLX',     # Netflix
    'MCD',      # McDonald's
    'DIS',      # Walt Disney
    'SAP',      # SAP (Germany)
    'LIN',      # Linde
    'AZN',      # AstraZeneca (UK)
    'ACN',      # Accenture
    'BHP',      # BHP Group (Australia)
    'INTC',     # Intel
    'PFE',      # Pfizer
    'VZ',       # Verizon
    'SHEL',     # Shell (UK)
    'T',        # AT&T
    'NKE',      # Nike
    'AMD',      # AMD
    'TMUS',     # T-Mobile US
]


start_date = '2015-01-01'
data = yf.download(tickers, period='30d', interval='1h', group_by='ticker')



returns = data.pct_change().dropna()


returns_close = returns.xs('Close', axis=1, level=1)
mean_returns = returns_close.mean() * 252  


cov_matrix = returns_close.cov() * 252

num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []


for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    weights_record.append(weights)
    port_return = np.sum(mean_returns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_vol
    results[0,i] = port_return
    results[1,i] = port_vol
    results[2,i] = sharpe


max_sharpe_ratio_id = np.argmax(results[2])

optimal_weights = weights_record[max_sharpe_ratio_id]



print("Portfolio with highest Sharpe Ratio:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")


print("\nExpected annual return: {:.2f}%".format(results[0, max_sharpe_ratio_id] * 100))
print("Annual volatility: {:.2f}%".format(results[1, max_sharpe_ratio_id] * 100))
print("Sharpe Ratio: {:.2f}".format(results[2, max_sharpe_ratio_id]))

plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.title('Portfolio Optimization')
plt.show()