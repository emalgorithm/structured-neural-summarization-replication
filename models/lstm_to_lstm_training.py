from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from models.lstm_encoder import LSTMEncoder
from models.lstm_decoder import LSTMDecoder

import torch
import torch.nn as nn
from torch import optim
import pickle
from sklearn.metrics import f1_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


# class Lang:
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {0: "SOS", 1: "EOS"}
#         self.n_words = 2  # Count SOS and EOS
#
#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)
#
#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1


class TokenLang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# def readLangs(lang1, lang2, reverse=False):
#     print("Reading lines...")
#
#     # Read the file and split into lines
#     lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
#         read().strip().split('\n')
#
#     # Split every line into pairs and normalize
#     pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
#
#     # Reverse pairs, make Lang instances
#     if reverse:
#         pairs = [list(reversed(p)) for p in pairs]
#         input_lang = Lang(lang2)
#         output_lang = Lang(lang1)
#     else:
#         input_lang = Lang(lang1)
#         output_lang = Lang(lang2)
#
#     return input_lang, output_lang, pairs


def read_tokens():
    data = pickle.load(open('../data/methods_tokens_data.pkl', 'rb'))
    methods_source = data['methods_source'][:1000]
    methods_names = data['methods_names'][:1000]

    pairs = [(methods_source[i], methods_names[i]) for i in range(len(methods_source))]
    return pairs

def prepare_tokens():
    lang = TokenLang('code')
    pairs = read_tokens()
    print("Read %s sentence pairs" % len(pairs))
    for pair in pairs:
        lang.addSentence(pair[0])
        lang.addSentence(pair[1])
    print("Counted words:")
    print(lang.name, lang.n_words)
    return lang, pairs


# def prepareData(lang1, lang2, reverse=False):
#     input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
#     print("Read %s sentence pairs" % len(pairs))
#     print("Trimmed to %s sentence pairs" % len(pairs))
#     print("Counting words...")
#     for pair in pairs:
#         input_lang.addSentence(pair[0])
#         output_lang.addSentence(pair[1])
#     print("Counted words:")
#     print(input_lang.name, input_lang.n_words)
#     print(output_lang.name, output_lang.n_words)
#     return input_lang, output_lang, pairs


# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# print(random.choice(pairs))
lang, pairs = prepare_tokens()
print(random.choice(pairs))


# def indexesFromSentence(lang, sentence):
#     return [lang.word2index[word] for word in sentence.split(' ')]

def indexesFromSentenceTokens(lang, sentence):
    return [lang.word2index[word] for word in sentence]


# def tensorFromSentence(lang, sentence):
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromSentenceTokens(lang, sentence):
    indexes = indexesFromSentenceTokens(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# def tensorsFromPair(pair):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     return (input_tensor, target_tensor)

def tensorsFromPairTokens(pair):
    input_tensor = tensorFromSentenceTokens(lang, pair[0])
    target_tensor = tensorFromSentenceTokens(lang, pair[1])
    return (input_tensor, target_tensor)


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

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        pred.append(decoder_output.argmax().item())
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, np.array(pred)


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.3):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    f1 = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPairTokens(random.choice(pairs))
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
        f1 += f1_score(target_tensor.numpy().reshape(-1), pred, average='micro')

        # print("Pred: {}".format(pred))
        # print("Target: {}".format(target_tensor.numpy()))
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
# encoder1 = LSTMEncoder(input_lang.n_words, hidden_size).to(device)
# attn_decoder1 = LSTMDecoder(hidden_size, output_lang.n_words).to(device)
encoder1 = LSTMEncoder(lang.n_words, hidden_size).to(device)
attn_decoder1 = LSTMDecoder(hidden_size, lang.n_words).to(device)
trainIters(encoder1, attn_decoder1, 75000, print_every=100)