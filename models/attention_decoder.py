import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)
        self.gru = nn.LSTM(hidden_size, hidden_size).to(device)
        self.out = nn.Linear(hidden_size, output_size).to(device)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attention = nn.Linear(hidden_size * 2, 1).to(device)

    def forward(self, input, hidden, encoder_hiddens):
        # encoder_hiddens has shape [batch_size, input_seq_len, hidden_dim]
        # input has shape [batch_size, 1, 1]
        # output has shape [batch_size, 1, hidden_size]
        output = self.embedding(input).view(1, 1, -1)

        # Attention
        hiddens = torch.cat((encoder_hiddens, hidden[0].repeat(1, encoder_hiddens.size(1), 1)),
                            dim=2)
        attention_coeff = self.attention(hiddens)
        att_hidden = torch.mm(torch.squeeze(encoder_hiddens).t(), torch.squeeze(attention_coeff,
                                                                        2).t()).view(1, 1, -1)
        # context = torch.cat((hidden[0], att_hidden), dim=2)
        context = att_hidden

        output = F.relu(output)
        output, hidden = self.gru(output, (hidden[0], context))
        output = self.softmax(self.out(output[0]))
        return output, hidden
