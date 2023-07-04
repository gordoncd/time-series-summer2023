import yfinance as yf
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Fetch the S&P500 constituents
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]

# Retrieve historical data for each ticker since 1991
start_date = '1991-01-01'
end_date = '2023-07-01'  # Set the desired end date
all_data = yf.download(tickers, start=start_date, end=end_date)['Close']

# Calculate the simple daily returns
returns = all_data.pct_change()

# Calculate the daily median returns
median_returns = returns.median(axis=1)

# Label the returns based on the median return of the day
labels = returns.gt(median_returns, axis=0).astype(int)

# Stride lengths for window operation
study_period_length = 1000  # Total length of each study period
training_period = 750  # Length of the training period
stride = study_period_length - training_period
sequence_length = 240  # Length of input sequence for training

tickers = ['AAPL']
# Prepare the windows for each stock
for ticker in tickers:
    if ticker in returns:
        # Get returns and labels
        ticker_returns = returns[ticker].dropna()
        ticker_labels = labels[ticker].loc[ticker_returns.index]

        # Generate windows of data
        for i in range(0, len(ticker_returns) - sequence_length + 1, stride):
            # Get the return and label windows
            returns_window = ticker_returns.iloc[i:i+sequence_length].values.reshape(-1, 1)
            labels_window = ticker_labels.iloc[i:i+sequence_length].values.reshape(-1, 1)

            # Save windows to .npy files
            np.save(f'data/{ticker}_{i}_returns.npy', returns_window)
            np.save(f'data/{ticker}_{i}_labels.npy', labels_window)