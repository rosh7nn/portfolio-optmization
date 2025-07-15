import pandas as pd
import matplotlib.pyplot as plt

# Full NIFTY 50 ticker list
NIFTY50_TICKERS = [
    'ADANIENT.NS','ADANIPORTS.NS','APOLLOHOSP.NS','ASIANPAINT.NS','AXISBANK.NS',
    'BAJAJ-AUTO.NS','BAJFINANCE.NS','BAJAJFINSV.NS','BEL.NS','BHARTIARTL.NS',
    'CIPLA.NS','COALINDIA.NS','DIVISLAB.NS','DRREDDY.NS','EICHERMOT.NS',
    'GRASIM.NS','HCLTECH.NS','HDFCBANK.NS','HDFCLIFE.NS','HEROMOTOCO.NS',
    'HINDALCO.NS','HINDUNILVR.NS','ICICIBANK.NS','INDUSINDBK.NS','INFY.NS',
    'ITC.NS','JIOFIN.NS','JSWSTEEL.NS','KOTAKBANK.NS','LT.NS',
    'M&M.NS','MARUTI.NS','NESTLEIND.NS','NTPC.NS','ONGC.NS',
    'POWERGRID.NS','RELIANCE.NS','SBILIFE.NS','SBIN.NS','SHRIRAMFIN.NS',
    'SUNPHARMA.NS','TATACONSUM.NS','TATAMOTORS.NS','TATASTEEL.NS','TCS.NS',
    'TECHM.NS','TITAN.NS','TRENT.NS','ULTRACEMCO.NS','WIPRO.NS'
]

# Create DataFrame with constant Start and End dates
data = {
    'Ticker': NIFTY50_TICKERS,
    'Start': ['2023-01-01'] * len(NIFTY50_TICKERS),
    'End': ['2025-07-01'] * len(NIFTY50_TICKERS)
}

df = pd.DataFrame(data)

# Clean ticker labels for y-axis
df['CleanTicker'] = df['Ticker'].str.replace('.NS', '', regex=False)

# Convert dates
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Plot setup
fig, ax = plt.subplots(figsize=(11, 14))

# Plot stock membership bars
for idx, row in df.iterrows():
    ax.barh(row['CleanTicker'], (row['End'] - row['Start']).days, left=row['Start'], color="#2ca02c")

# Formatting
ax.set_xlabel('Date')
ax.set_ylabel('Ticker')
ax.set_title('NIFTY 50 Stocks Considered in the Study (2023–2025)', fontsize=14)
ax.invert_yaxis()
fig.autofmt_xdate()

# Save high-res PNG
plt.tight_layout()
plt.savefig('nifty50_stocks_considered_timeline.png', dpi=300)
plt.show()
