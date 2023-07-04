import yfinance as yf
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Retrieve S&P 500 constituents
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table', {'class': 'wikitable sortable'})

tickers = []
for row in table.find_all('tr')[1:]:
    ticker = row.find_all('td')[0].text.strip()
    tickers.append(ticker)

tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'GOOGL', 'FB', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'INTC', 'CMCSA', 'NFLX']

# Join all tickers into a single string with spaces in between
tickers_string = ' '.join(tickers)

# Define the study period parameters
study_period_length = 1000
training_period = 750
sequence_length = 240
stride = 250  # Rolling forward by 250 days

# Retrieve historical data for all tickers at once since 1991
start_date = '1991-01-01'
end_date = '2023-07-01'

try:
    all_data = yf.download(tickers_string, start=start_date, end=end_date)
except Exception as e:
    print(f"Failed to download data: {str(e)}")

# Extract 'Adj Close' columns
print(all_data.head())
all_data = all_data['Adj Close']
print(all_data.head())

# Create a new DataFrame to store simple returns with single-level columns
with_simple_returns = pd.DataFrame({ticker: all_data[ticker].pct_change() for ticker in tickers})

# Calculate the cross-sectional median
with_simple_returns['Cross-sectional Median'] = with_simple_returns.median(axis=1)
with_simple_returns = with_simple_returns.tail(-1)
print(with_simple_returns.head())

# Process each study period
study_periods = []
labels = []

for i in tqdm(range(0, len(with_simple_returns) - study_period_length + 1, stride), desc="Generating Study Periods"):
    period_data = with_simple_returns.iloc[i:i + study_period_length].copy()

    # Drop columns with null values
    period_data = period_data.dropna(axis=1)

    # Create overlapping sequences for the training period
    training_data = period_data
    median_to_label = period_data['Cross-sectional Median'].values

    label = np.where(training_data.values < median_to_label[:, np.newaxis], 0, 1)

    sequences = [training_data.values[j:j + sequence_length] for j in range(0, len(training_data) - stride, 1)]
    label = [label[j:j + sequence_length] for j in range(0, len(training_data) - stride, 1)]

    # Add to the study periods list
    study_periods.append(sequences)
    labels.append(label)

#loop through study period and we want to reshape along the 0th axis
study_periods_array = None
labels_array = None

study_periods_array = np.concatenate([np.array(array) for array in study_periods if array is not None], axis=2)

labels_array = np.concatenate([np.array(array) for array in labels if array is not None], axis=2)
# Check if there are at least three dimensions before attempting to swap axes
if len(study_periods_array.shape) >= 3:
    study_periods_array = study_periods_array.swapaxes(0, 2)
    study_periods_array = study_periods_array.swapaxes(1, 2)
else:
    print("Study periods array has less than three dimensions. Cannot swap axes.")
    # Add some error handling or debugging here if needed

medians = np.array(labels_array)
medians = medians.swapaxes(0, 2)
medians = medians.swapaxes(1, 2)

# Now we want to stack the last dimension into the array along the 0th axis
print(medians.shape)
print(study_periods_array.shape)

print(medians.shape)
print(study_periods_array.shape)
# Save the formatted data and labels

np.save("data/test-sp500-simple-return-periodized.npy", study_periods_array)
np.save("data/test-sp500-simple-return-labels.npy", medians)
