import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from train import train
from test import test
from model import LSTMNetwork

# Define the network configuration
input_size = 1
hidden_size = 25
num_layers = 1
output_size = 2
dropout = 0.1

# Create an instance of the LSTMNetwork
net = LSTMNetwork(input_size, hidden_size, num_layers, output_size, dropout)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters())

# Set the maximum number of training epochs and early stopping patience
max_epochs = 1000
patience = 10

# Load the training, validation, and test data from CSV files
train_data = pd.read_csv("data/train_data.csv")
val_data = pd.read_csv("data/val_data.csv")
test_data = pd.read_csv("data/test_data.csv")

# Create the train_loader, val_loader, and test_loader
train_inputs = torch.tensor(train_data["Sequence"].values).unsqueeze(dim=2).float()
train_labels = torch.tensor(train_data["Label"].values).long()
train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_inputs = torch.tensor(val_data["Sequence"].values).unsqueeze(dim=2).float()
val_labels = torch.tensor(val_data["Label"].values).long()
val_dataset = TensorDataset(val_inputs, val_labels)
val_loader = DataLoader(val_dataset, batch_size=32)

test_inputs = torch.tensor(test_data["Sequence"].values).unsqueeze(dim=2).float()
test_labels = torch.tensor(test_data["Label"].values).long()
test_dataset = TensorDataset(test_inputs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32)

# Train the model
best_model_state_dict = train(net, train_loader, val_loader, criterion, optimizer, max_epochs, patience)

# Save the best model state dictionary to a file
torch.save(best_model_state_dict, "best_model.pth")

# Load the saved model
loaded_model = LSTMNetwork(input_size, hidden_size, num_layers, output_size, dropout)
loaded_model.load_state_dict(torch.load("best_model.pth"))

# Test the model
test(loaded_model, test_loader, criterion)
