import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, device, attention=False, pointer_network=False):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)
        self.gru = nn.LSTM(hidden_size, hidden_size).to(device)
        self.out = nn.Linear(hidden_size, output_size).to(device)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = attention
        self.pointer_network = pointer_network
        self.attention_layer = nn.Linear(hidden_size * 2, 1).to(device)
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size).to(device)
        self.device = device

    def forward(self, input, hidden, encoder_hiddens, input_seq=None):
        # encoder_hiddens has shape [batch_size, seq_len, hidden_dim]
        output = self.embedding(input).view(1, 1, -1)

        if self.attention:
            # Create a matrix of shape [batch_size, seq_len, 2 * hidden_dim] where the last
            # dimension is a concatenation of the ith encoder hidden state and the current decoder
            # hidden
            hiddens = torch.cat((encoder_hiddens, hidden[0].repeat(1, encoder_hiddens.size(1), 1)),
                                dim=2)

            # attention_coeff has shape [seq_len] and contains the attention coeffiecients for
            # each encoder hidden state
            # attention_coeff has shape [batch_size, seq_len, 1]
            attention_coeff = self.attention_layer(hiddens)
            attention_coeff = torch.squeeze(attention_coeff, dim=2)
            attention_coeff = torch.squeeze(attention_coeff, dim=0)
            attention_coeff = F.softmax(attention_coeff, dim=0)

            # Make encoder_hiddens of shape [hidden_dim, seq_len] as long as batch size is 1
            encoder_hiddens = torch.squeeze(encoder_hiddens, dim=0).t()

            context = torch.matmul(encoder_hiddens, attention_coeff).view(1, 1, -1)
            output = torch.cat((output, context), 2)
            output = self.attention_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden
