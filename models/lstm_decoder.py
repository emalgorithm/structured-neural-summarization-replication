import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, device, attention=False):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)
        self.gru = nn.LSTM(hidden_size, hidden_size).to(device)
        self.out = nn.Linear(hidden_size, output_size).to(device)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = attention
        self.attention_layer = nn.Linear(hidden_size * 2, 1).to(device)
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size).to(device)
        self.device = device

    def forward(self, input, hidden, encoder_hiddens):
        # encoder_hiddens has shape [batch_size, seq_len, hidden_dim]
        output = self.embedding(input).view(1, 1, -1)

        if self.attention:
            hiddens = torch.cat((encoder_hiddens, hidden[0].repeat(1, encoder_hiddens.size(1), 1)),
                                dim=2)
            attention_coeff = self.attention_layer(hiddens)
            context = torch.mm(torch.squeeze(encoder_hiddens, dim=0).t(), torch.squeeze(
                attention_coeff, 2).t()).view(1, 1, -1)
            output = torch.cat((output, context), 2)
            output = self.attention_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden
