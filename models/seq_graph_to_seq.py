import torch
import torch.nn as nn


class SeqGraph2Seq(nn.Module):
    def __init__(self, seq_encoder, graph_encoder, seq_decoder, device):
        super().__init__()

        self.seq_encoder = seq_encoder
        self.graph_encoder = graph_encoder
        self.seq_decoder = seq_decoder
        self.device = device

        assert seq_encoder.hidden_size == seq_decoder.hidden_size, "Hidden dimensions of encoder and decoder " \
                                                    "must be equal!"

    def forward(self, sequence, adj, target):
        batch_size = 1
        max_len = target.shape[0]
        target_vocab_size = self.seq_decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, target_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        # output has dimension [seq_len, batch, hidden_size]
        output, hidden = self.seq_encoder(sequence)

        # graph_hidden has shape [1, 1, hidden_size] and contains a graph representation
        n_nodes = adj.size(0)
        n_tokens = sequence.size(0)
        x = torch.zeros(n_nodes, output.size(2))
        x[:n_tokens, :] = output.view(output.size(1), output.size(2))
        graph_hidden = self.graph_encoder(x=x, adj=adj)

        # first input to the decoder is the <sos> tokens
        input = torch.tensor([[0]], device=self.device)
        hidden = (graph_hidden.view(1, 1, graph_hidden.size(0)), hidden[1])

        for t in range(1, max_len):
            output, hidden = self.seq_decoder(input, hidden)
            outputs[t] = output
            top1 = output.max(1)[1]
            input = top1

        return outputs
