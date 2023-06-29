# Time Series Analysis/Prediction with Keras (training wheels)

This code performs time series analysis and prediction using Keras, a high-level neural networks API. The code utilizes historical stock data from the 'GOOGL' ticker symbol obtained through the Yahoo Finance API. The goal is to predict the next day's stock price movement.

## Dependencies
The code requires the following dependencies:
- tensorflow
- keras
- yfinance
- pandas
- numpy
- matplotlib
- pandas_ta
- sklearn

You may need to install the 'pandas_ta' package using the command `! pip install pandas_ta` in the code.

## Data Preparation
1. The code fetches historical stock data for the 'GOOGL' ticker symbol using the Yahoo Finance API. The data is fetched for the period from '2018-01-01' to '2023-01-01'.
2. Various technical indicators are calculated and added to the data, including:
   - RSI (Relative Strength Index) with a length of 15
   - EMA (Exponential Moving Average) with lengths of 20, 100, and 150
   - Other technical indicators can be added as needed
3. The target variable is calculated as the difference between the adjusted closing price ('Adj Close') and the opening price ('Open') of each day. The target variable is then shifted by one day to align it with the corresponding input data.
4. The target variable is transformed into a binary class variable, where 1 represents a positive target (price increase) and 0 represents a negative target (price decrease or no change).
5. The 'Adj Close' of the next day is stored as a separate target variable 'TargetNextClose' for evaluation purposes.
6. Any rows with missing values (NaN) are dropped from the dataset.
7. The 'Volume', 'Close', and 'Date' columns are dropped from the dataset.
8. The remaining dataset is scaled using the MinMaxScaler to normalize the values between 0 and 1.

## Model Training and Testing
1. The input data 'X' is prepared by sliding a window of 'backcandles' (30) previous days' data over the dataset. This creates a 3D array where the first dimension represents the number of samples, the second dimension represents the time steps (backcandles), and the third dimension represents the features (8 technical indicators).
2. The target variable 'y' is reshaped into a 2D array.
3. The data is split into training and testing sets using an 80:20 ratio. The split threshold is calculated based on the length of 'X'.
4. The hyperparameters for the LSTM (Long Short-Term Memory) model are defined:
   - lstm_size: The number of LSTM units in the first layer (150)
   - batch_size: The number of samples per gradient update during training (15)
   - epochs: The number of times the entire dataset is passed forward and backward through the model during training (30)
5. The LSTM model is defined using the Keras functional API. It consists of an LSTM layer followed by a Dense layer and an output Activation layer.
6. The Adam optimizer is used with mean squared error (MSE) loss for compilation of the model.
7. The model is trained using the training data. Validation split of 0.1 is used to evaluate the model performance during training.
8. The trained model is used to make predictions on the test data 'X_test'.
9. The predicted values 'y_pred' are plotted against the actual values 'y_test' using matplotlib.

## Room for Adjustments and Performance
1. Technical Indicators: Additional technical indicators can be added to

 the code by using the 'pandas_ta' library. These indicators can provide more information and potentially improve the model's performance. You can explore different indicators and their parameters to find the ones that work best for the specific stock or time series data.
2. Model Architecture: The current model architecture consists of a single LSTM layer followed by a Dense layer. You can adjust the architecture by adding more LSTM layers, fully connected layers, or other types of layers (e.g., convolutional layers) to potentially capture more complex patterns in the data.
3. Hyperparameters: The hyperparameters of the model, such as the number of LSTM units ('lstm_size'), batch size ('batch_size'), and the number of epochs ('epochs'), can be adjusted to improve the model's performance. Tuning these hyperparameters can help in finding the right balance between underfitting and overfitting.
4. Data Preprocessing: The code currently uses the MinMaxScaler to normalize the data. You can explore other scaling techniques or preprocessing methods to handle outliers or skewness in the data, which may improve the model's performance.
5. Model Evaluation: The code currently uses mean squared error (MSE) as the loss function for training the model. You can experiment with different loss functions or evaluation metrics to assess the model's performance more effectively, depending on the specific requirements of your prediction task.
