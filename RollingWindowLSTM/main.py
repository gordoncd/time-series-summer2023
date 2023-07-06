import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from train import train
from test import test
from model import LSTMNetwork

# Define the network configuration
input_size = 1
hidden_size = 25
output_size = 2

# Create an instance of the LSTMNetwork
net = LSTMNetwork(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters())

# Set the maximum number of training epochs and early stopping patience
max_epochs = 1000
patience = 10
print(os.getcwd())
print(os.listdir())
#load in the data 
X = np.load('data/test-sp500-simple-return-periodized.npy')
y = np.load("data/test-sp500-simple-return-labels.npy")

print(X.shape, y.shape)
y = y.reshape(-1, )

# Calculate the number of samples for each split
num_samples = len(X)
num_train = int(0.8 * num_samples)
num_val = int(0.1 * num_train)

# Split the data
X_train = X[:num_train]
y_train = y[:num_train]
X_val = X[num_train:num_train+num_val]
y_val = y[num_train:num_train+num_val]
X_test = X[num_train+num_val:]
y_test = y[num_train+num_val:]


# Create the train_loader, val_loader, and test_loader
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()), batch_size=32, shuffle=False)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), batch_size=32, shuffle=False)


# Train the model
best_model_state_dict = train(net, train_loader, val_loader, criterion, optimizer, max_epochs, patience)

# Save the best model state dictionary to a file
torch.save(best_model_state_dict, "best_model.pth")

# Load the saved model
loaded_model = LSTMNetwork(input_size, hidden_size, output_size)
loaded_model.load_state_dict(torch.load("best_model.pth"))

# Test the model
test(loaded_model, test_loader, criterion)
