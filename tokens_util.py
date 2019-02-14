from __future__ import unicode_literals, print_function, division
from io import open
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pylab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)


class TokenLang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.SOS_token = 0
        self.EOS_token = 1

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

    def to_tokens(self, idxs):
        return np.array([self.index2word[idx] for idx in idxs])


def read_tokens():
    data = pickle.load(open('data/methods_tokens_data.pkl', 'rb'))
    methods_source = data['methods_source']
    methods_names = data['methods_names']

    pairs = [(methods_source[i], methods_names[i]) for i in range(len(methods_source))]
    np.random.shuffle(pairs)

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


def indexes_from_sentence_tokens(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensor_from_sentence_tokens(lang, sentence):
    indexes = indexes_from_sentence_tokens(lang, sentence)
    indexes.append(lang.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair_tokens(pair, lang):
    input_tensor = tensor_from_sentence_tokens(lang, pair[0])
    target_tensor = tensor_from_sentence_tokens(lang, pair[1])
    return input_tensor, target_tensor


def plot_loss(train_losses, val_losses, file_path='plots/loss.jpg'):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(('train loss', 'validation loss'), loc='upper right')
    plt.title('Losses during training of LSTM->LSTM Model')
    plt.xlabel('#epochs')
    plt.ylabel('cross-entropy loss')
    plt.show()
    pylab.savefig(file_path)
