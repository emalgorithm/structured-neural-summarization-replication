import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder " \
                                                    "must be equal!"

    def forward(self, sequence, target):
        batch_size = 1
        max_len = target.shape[0]
        target_vocab_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, target_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        output, hidden = self.encoder(sequence)

        # first input to the decoder is the <sos> tokens
        input = torch.tensor([[0]], device=self.device)

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            top1 = output.max(1)[1]
            input = top1

        return outputs

