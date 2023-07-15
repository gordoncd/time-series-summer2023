import yfinance as yf
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

def download_smp_csv(tickers, start_date, end_date, dest):
    """
    Downloads stock price data for the given tickers, from start_date to end_date. It is part of the data collection step.
    :param tickers: A list of company tickers.
    :param start_date: A string representing the start date in 'YYYY-MM-DD' format. For example, '1991-01-01'.
    :param end_date: A string representing the end date in 'YYYY-MM-DD' format. For example, '2023-07-01'.
    :return: A pandas DataFrame of 'Adjusted Close' prices of the stocks, or None if data download fails.
    """
    tickers_string = ' '.join(tickers)
    try:
        all_data = yf.download(tickers_string, start=start_date, end=end_date)
    except Exception as e:
        print(f"Failed to download data: {str(e)}")

    # save all data to csv
    all_data['Adj Close'].to_csv(dest)

    
def get_sp500_tickers():
    """
    Retrieves the list of S&P 500 companies from the corresponding Wikipedia page. This list is used to download stock data.
    :return: A list of tickers of all companies listed on the S&P 500.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[0].text.strip()
        tickers.append(ticker)
    response.close()
    return tickers

def get_data(tickers, start_date, end_date):
    """
    Downloads stock price data for the given tickers, from start_date to end_date. It is part of the data collection step.
    :param tickers: A list of company tickers.
    :param start_date: A string representing the start date in 'YYYY-MM-DD' format. For example, '1991-01-01'.
    :param end_date: A string representing the end date in 'YYYY-MM-DD' format. For example, '2023-07-01'.
    :return: A pandas DataFrame of 'Adjusted Close' prices of the stocks, or None if data download fails.
    """
    tickers_string = ' '.join(tickers)
    try:
        all_data = yf.download(tickers_string, start=start_date, end=end_date)
    except Exception as e:
        print(f"Failed to download data: {str(e)}")
        return None
    return all_data['Adj Close']

def create_simple_returns(all_data, tickers):
    """
    Computes the simple returns for each stock and calculates the cross-sectional median return. This transformation is required to feed the data to the model.
    :param all_data: A pandas DataFrame containing 'Adjusted Close' prices of the stocks.
    :param tickers: A list of company tickers.
    :return: A pandas DataFrame of simple returns and cross-sectional median for each day.
    """
    with_simple_returns = pd.DataFrame({ticker: all_data[ticker].pct_change() for ticker in tickers})
    with_simple_returns['Cross-sectional Median'] = with_simple_returns.median(axis=1)
    return with_simple_returns.tail(-1)

def create_study_periods(with_simple_returns, study_period_length, sequence_length, stride, include_median = True):
    """
    Generates study periods and labels for each period. Each study period is a sequence of historical returns and the label indicates if the return is below or above the median.
    :param with_simple_returns: A pandas DataFrame of simple returns and cross-sectional median.
    :param study_period_length: An integer specifying the length of a study period, such as 1000 days.
    :param sequence_length: An integer specifying the sequence length within each study period, such as 240 days.
    :param stride: An integer specifying the number of days to roll forward, such as 250 days.
    :param include_median: A boolean specifying whether to include cross-sectional median.
    :return: A tuple of lists (study_periods, labels) if include_median=False, or (study_periods, labels, median_arr) if include_median=True.
    """
    study_periods = []
    labels = []
    median_arr = []
    trues = []

    for i in tqdm(range(0, len(with_simple_returns) - study_period_length + 1, stride), desc="Generating Study Periods"):
        period_data = with_simple_returns.iloc[i:i + study_period_length].copy()
        period_data = period_data.dropna(axis=1)
        training_data = period_data
        median_to_label = period_data['Cross-sectional Median'].values
        label = np.where(training_data.values < median_to_label[:, np.newaxis], 0, 1)
        sequences = [training_data.values[j:j + sequence_length] for j in range(0, len(training_data) - sequence_length-1, 1)]
        label = [label[1+j + sequence_length] for j in range(0, len(training_data) - sequence_length-1, 1)]
        true  = [training_data.values[1+j+sequence_length] for j in range(0, len(training_data) - sequence_length-1, 1)]
        if include_median:
            median_formatted = [median_to_label[1+j + sequence_length] for j in range(0, len(training_data) - sequence_length-1, 1)]
            median_formatted = np.array(median_formatted)
            median_arr.append(median_formatted)
        trues.append(true)
        label = np.array(label)
        study_periods.append(sequences)
        labels.append(label)
    if include_median:
        return study_periods, labels, median_arr, trues
    else:
        return study_periods, labels, trues

def reshape_arrays(study_periods, labels, medians = None):
    """
    Reshapes the study periods and labels into the format required for training a machine learning model. This step is necessary to match the input shape requirements of many machine learning models.
    :param study_periods: A list of study periods.
    :param labels: A list of labels corresponding to each study period.
    :return: A tuple of numpy arrays (reshaped study periods, reshaped labels).
    """
    study_periods_array = np.concatenate(
        [np.array(array) for array in tqdm(study_periods, desc="Study Periods") if array is not None],
        axis=2
    )

    labels_array = np.concatenate(
        [np.array(array) for array in tqdm(labels, desc="Labels") if array is not None],
        axis=1
    )

    if medians is not None: 
        medians_array = np.concatenate(
            [np.array(array) for array in tqdm(medians, desc="Medians") if array is not None],
            axis=0
        )
        medians_array = np.array(medians_array)
        medians_array = np.reshape(medians_array, (medians_array.shape[0], 1))
    

    if len(study_periods_array.shape) >= 3:
        study_periods_array = study_periods_array.swapaxes(0, 2)
        study_periods_array = study_periods_array.swapaxes(1, 2)
    else:
        print("Study periods array has less than three dimensions. Cannot swap axes.")

    labels_array = np.array(labels_array)
    labels_array = labels_array.swapaxes(0,1)
    labels_array = np.reshape(labels_array, (-1, 1))
    study_periods_array = np.reshape(study_periods_array, (-1, 240, 1))
    if medians is not None:
        return study_periods_array, labels_array, medians_array

    return study_periods_array, labels_array

def reshape_arrays_comparison(labels, trues):
    labels_array = np.concatenate(
    [np.array(array) for array in tqdm(labels, desc="Labels") if array is not None],
    axis=1)
    trues_array = np.concatenate(
    [np.array(array) for array in tqdm(trues, desc="Trues") if array is not None], axis = 1)

    labels_array = np.swapaxes(labels_array, 0, 1)
    trues_array = np.swapaxes(trues_array, 0, 1)

    return labels_array, trues_array

def save_data(study_periods_array, labels_array):
    """
    Saves the study periods and labels to numpy files. This allows the data to be loaded later for analysis or model training.
    :param study_periods_array: A numpy array of reshaped study periods.
    :param labels_array: A numpy array of reshaped labels.
    """
    np.save("data/sp500-simple-return-periodized.npy", study_periods_array)
    np.save("data/sp500-simple-return-labels.npy", labels_array)

def main():
    """
    The main function of the program. Coordinates all the steps: downloading stock price data, calculating returns, generating study periods and labels, reshaping the arrays, and saving them to disk. It sets specific values for the study period length (1000), sequence length (240), and stride (250). It also specifies a fixed list of tickers ['AAPL', 'MMM', 'AMZN', 'MSFT', 'TSLA'] for demonstration purposes.
    """
    study_period_length = 1000
    sequence_length = 240
    stride = 250  # Rolling forward by 250 days
    #load data from csv
    all_data = pd.read_csv('data/sp500-all-data.csv', index_col=0)
    #grab tickers from column names
    tickers = all_data.columns.values
    with_simple_returns = create_simple_returns(all_data, tickers)
    study_periods, labels, crossec_meds, __ = create_study_periods(with_simple_returns, study_period_length, sequence_length, stride, True)
    
    study_periods_array, labels_array, medians_array= reshape_arrays(study_periods, labels, crossec_meds)
    print(study_periods_array.shape)
    print(labels_array.shape)
    print(medians_array.shape)
    save_data(study_periods_array, labels_array)

def comparison():
    study_period_length = 1000
    sequence_length = 240
    stride = 250

    # Load data from csv
    all_data = pd.read_csv('data/sp500-all-data.csv', index_col=0)
    tickers = all_data.columns.values
    with_simple_returns = create_simple_returns(all_data, tickers)
    study_periods, labels, crossec_meds, trues = create_study_periods(with_simple_returns, study_period_length, sequence_length, stride, True)

    # Check array shapes
    print(np.array(labels[0]).shape, np.array(trues[0]).shape, np.array(crossec_meds).shape)
    
    num_periods = len(labels)
    medians = np.array(crossec_meds)

    labels_array, trues_array = [], []

    for period in range(num_periods):
        label_period = np.array(labels[period])
        true_period = np.array(trues[period])
        labels_array.append(label_period)
        trues_array.append(true_period)
        
        for step in range(labels[period].shape[0]):
            days_median = medians[period][step]
            
            for stock in range(labels[0].shape[1]):  
                if true_period[step, stock] > days_median and label_period[step, stock] != 1:
                    print("problem")
                if true_period[step, stock] < days_median and label_period[step, stock] != 0:
                    print("problem")

    labels_array, trues_array = reshape_arrays_comparison(labels_array, trues_array)

    print(labels_array.shape, trues_array.shape)
    #save arrays 
    np.save("data/sp500-labels.npy", labels_array)
    np.save("data/sp500-trues.npy", trues_array)
    np.save('data/sp500-medians.npy', medians)

                


if __name__ == "__main__":
    comparison()
   


