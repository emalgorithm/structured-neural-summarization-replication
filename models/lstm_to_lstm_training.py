from __future__ import unicode_literals, print_function, division
import random
from models.lstm_encoder import LSTMEncoder
from models.lstm_decoder import LSTMDecoder
from tokens_util import prepare_tokens, tensors_from_pair_tokens

import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import f1_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lang, pairs = prepare_tokens()
print(random.choice(pairs))

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=7100):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0
    pred = []

    encoder_outputs, encoder_hidden = encoder(input_tensor.view(-1), encoder_hidden)

    decoder_input = torch.tensor([[lang.SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        pred.append(decoder_output.argmax().item())
        if decoder_input.item() == lang.EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, np.array(pred)


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.1):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    f1 = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair_tokens(random.choice(pairs), lang)
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss, pred = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        y_true = target_tensor.numpy().reshape(-1)

        if len(y_true) < len(pred):
            y_true = np.pad(y_true, (0, len(pred) - len(y_true)), mode='constant')
        else:
            pred = np.pad(pred, (0, len(y_true) - len(pred)), mode='constant')

        f1 += f1_score(y_true, pred, average='micro')

        # print("Pred: {}".format(lang.to_tokens(pred)))
        # print("Target: {}".format(lang.to_tokens(target_tensor.numpy().reshape(-1))))
        # print()


        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))
            print('f1_score: {}'.format(f1 / iter))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


hidden_size = 256
encoder1 = LSTMEncoder(lang.n_words, hidden_size).to(device)
attn_decoder1 = LSTMDecoder(hidden_size, lang.n_words).to(device)
trainIters(encoder1, attn_decoder1, 75000, print_every=10)