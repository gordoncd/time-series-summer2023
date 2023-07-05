
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (1, batch_size, hidden_size)
        h_n = h_n.squeeze(0)
        # h_n shape: (batch_size, hidden_size)
        out = self.dropout(h_n)
        out = self.fc(out)
        out = self.softmax(out)
        return out