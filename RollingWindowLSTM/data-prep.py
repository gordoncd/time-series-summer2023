import yfinance as yf
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Step 1: Retrieve S&P 500 constituents
sp500 = yf.Tickers('^GSPC')
constituents = sp500.tickers
print(constituents)

# Send a GET request to the Wikipedia page for S&P 500
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table containing the S&P 500 ticker data
table = soup.find('table', {'class': 'wikitable sortable'})

# Initialize an empty list to store tickers
tickers = []

# Extract ticker symbols from the table rows
for row in table.find_all('tr')[1:]:
    ticker = row.find_all('td')[0].text.strip()
    tickers.append(ticker)

# Define the study period parameters
study_period_length = 1000  # Total length of each study period (training + trading)
training_period = 750  # Length of the training period
sequence_length = 240  # Length of input sequence for training

# Retrieve historical data for each ticker since 1991
start_date = '1991-01-01'
end_date = '2023-07-01'  # Set the desired end date

# Initialize an empty list to store the study periods
study_periods = []
tickers = ['AAPL']
# Retrieve data for each ticker
formatted_data = pd.DataFrame(columns=['Ticker', 'data'])
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Ticker'] = ticker  # Add a column to indicate the ticker symbol

    # Calculate the simple daily returns
    data['Simple Return'] = (data['Close'] / data['Close'].shift(1)) - 1

    # Split the data into study periods with overlapping sequences
    num_periods = len(data) // study_period_length
    data_splits = [data[i * study_period_length:(i + 1) * study_period_length].copy() for i in range(num_periods)]

    # Process each study period
    for study_period in data_splits:
        # Extract the training data and create overlapping sequences
        if len(study_period) >= training_period:
            training_data = study_period
            sequences = np.lib.stride_tricks.sliding_window_view(training_data['Simple Return'].values,
                                                                    (sequence_length,))
            study_period_df = pd.DataFrame(sequences.T, columns=[i for i in range(sequences.shape[0])])
            study_periods.append(study_period_df)

    # Concatenate the study periods into a single DataFrame
    study_periods_data = pd.concat(study_periods, axis=1)

    # Print the first few rows of the study periods data
    print(study_periods_data.head())

    # now we want to create numpy array for study_periods_data
    numpy_study_periods_data = study_periods_data.to_numpy()  # remember that columns are days and rows are timesteps here
    print(numpy_study_periods_data.shape)
    formatted_data = pd.concat([formatted_data, pd.DataFrame({'Ticker': ticker, 'data': numpy_study_periods_data})], ignore_index=True)

print(formatted_data)

