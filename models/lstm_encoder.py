import torch.nn as nn
import torch


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size).to(device)
        self.gru = nn.LSTM(hidden_size, hidden_size, batch_first=True).to(device)
        self.device = device

    def forward(self, input):
        hidden = self.init_hidden()
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size).to(self.device),
                torch.zeros(1, 1, self.hidden_size).to(self.device))
