# Predicting Direction of Input Stock from Sequence of n Prior Days. LSTM with Rolling Window Approach 
<p style="text-align: center;">Gordon Doore </p>
<p style="text-align: center;">Colby College '25 </p>
<p style="text-align: center;">Summer 2023 Independent Project </p>

# Project Overview


## Table of Contents

1. [Description](#description)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Description<a name="description"></a>

### Data Generation:

#### generate_labels.py

1. Load data from the CSV file:
   - The `all_data` variable is assigned the value of the data loaded from the CSV file using `pd.read_csv()`.
   - This `all_data` DataFrame contains the 'Adjusted Close' prices of all of the stocks.

2. Calculate simple returns and cross-sectional medians:
   - `create_simple_returns()` function is called, passing `all_data` and the list of tickers.
   - The function computes the simple returns for each stock and calculates the cross-sectional median return for each day for the entire S&P 500.
   - The result is assigned to `with_simple_returns` DataFrame, which contains the simple returns and cross-sectional medians across the entire dataset.

3. Generate study periods and labels:
   - `create_study_periods()` function is called, passing `with_simple_returns` and other parameters such as study period length, sequence length, stride, and the option to include the median.
   - The function generates study periods, labels, cross-sectional medians, and actual returns.
   - The study periods are assigned to the `study_periods` list, the labels to the `labels` list, the cross-sectional medians to the `crossec_meds` list, and the actual returns to the `trues` list.

4. Reshape the arrays:
   - `reshape_arrays()` function is called, passing `study_periods`, `labels`, and `crossec_meds` lists.
   - The function reshapes the study periods, labels, and cross-sectional medians arrays into the required format for training a machine learning model.
   - The reshaped study periods are assigned to the `study_periods_array`, the reshaped labels to the `labels_array`, and the reshaped cross-sectional medians to the `medians_array`.

5. Print the shapes of the arrays:
   - The shapes of the `study_periods_array`, `labels_array`, and `medians_array` are printed using `print()` statements.

6. Save the reshaped arrays to disk:
   - `save_data()` function is called, passing the `study_periods_array` and `labels_array`.
   - The function saves the study periods and labels arrays to numpy files.

The operations here are very memory intensive and can be extremely slow if your machine has limited memory.

#### Data processing from model:

Here, we do a final reshaping: 
```python
# Now we get our data as an array
y = np.load("/content/sp500-simple-return-labels.npy", mmap_mode='r')
y = y.reshape(-1,)
X = np.load("/content/sp500-simple-return-periodized.npy", mmap_mode='r')
```

We split as 80% train, 10% validation, 10% test.
because we have ~ 8 million samples, we can use this breakdown because the test set will still be non-trivial in size and distinction from the training data.

##### Data Loader:
our model is looking for our data to come batched, so we use PyTorch's DataLoader object

we define our datasets as follows: 

```python
train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float().unsqueeze(1))
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float().unsqueeze(1))
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
```

Here we can modify the batch size,and this is still being experimented on.
I shuffled the training data to ensure against overtraining (sliding window has highly similar data next to eachother)
The data is then unsqueezed to be accepted into the model's criterion (BCELoss)

The data is now acceptable for entry into the model.

### Model Architecture:

![Diagram of model](imgs/model_diagram.png)

Our model has the input layer which then moves into the LSTM layer.  Next, the output of the LSTM is normalized using batch normalization.  In this case, we have our batchsize set to 2. The output of this normalization then has 10% dropout performed to avoid overfitting and a ReLU afterward to help reduce linearity in the data to help the model learn better. Next, the output of ReLU is entered into a fully connected layer of hidden size 1, thus reducing the dimensionality of the output to our binary classification. Finally, this is activated with sigmoid, which is the best output activation for binary classification problems.

### Model Training Loop:

The train() function trains the model by iterating over epochs, performing forward and backward passes for each batch, calculating training and validation losses, and updating the model's parameters using an optimizer. It implements early stopping based on the validation loss and returns the state dictionary of the model with the best validation loss.

I use Binary Cross Entropy Loss as our criterion and an RMSprop optimizer.  This follows Krauss and Fischer, 2018.

The rest of the training hyperparameters are still in progress and we do not have an effective combination of parameters.

### Testing: 

In progress


## Dependencies<a name="dependencies"></a>

The project has the following dependencies:

- [Dependency 1](link): [Brief description or purpose of the dependency].
- [Dependency 2](link): [Brief description or purpose of the dependency].
- [Dependency 3](link): [Brief description or purpose of the dependency].
- ...

Please make sure to install or set up these dependencies before using the project.

## Installation<a name="installation"></a>

To install the project, follow these steps:

1. Clone the repository: `git clone <repository_url>`
2. Navigate to the project directory: `cd project_directory`
3. Install the required dependencies: `npm install` or `pip install -r requirements.txt`

## Usage<a name="usage"></a>

To use the project, follow these steps:

1. [Provide step-by-step instructions on how to use the project]
2. [Include examples or code snippets if necessary]

## Contributing<a name="contributing"></a>

Contributions to this project are welcome. To contribute, please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b new_branch`
3. Make your changes and commit them: `git commit -am 'Add feature'`
4. Push the changes to your forked repository: `git push origin new_branch`
5. Submit a pull request detailing your changes

## License<a name="license"></a>

[Specify the license under which the project is distributed. For example, MIT, Apache, etc.]

Please refer to the LICENSE file for more information.

Feel free to reach out if you have any questions or need further assistance.

Happy coding!

