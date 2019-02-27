import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention=False):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)
        self.gru = nn.LSTM(hidden_size, hidden_size).to(device)
        self.out = nn.Linear(hidden_size, output_size).to(device)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = attention
        self.attention_layer = nn.Linear(hidden_size * 2, 1).to(device)

    def forward(self, input, hidden, encoder_hiddens):
        output = self.embedding(input).view(1, 1, -1)

        if self.attention:
            hiddens = torch.cat((encoder_hiddens, hidden[0].repeat(1, encoder_hiddens.size(1), 1)),
                                dim=2)
            attention_coeff = self.attention_layer(hiddens)
            context = torch.mm(torch.squeeze(encoder_hiddens).t(), torch.squeeze(attention_coeff,
                                                                            2).t()).view(1, 1, -1)
            # At the moment pass attention as cell state
            hidden = (hidden[0], context)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden
