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
    """
    Language of all tokens in a dataset.
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.SOS_token = 0
        self.EOS_token = 1

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def to_tokens(self, idxs):
        return np.array([self.index2word[idx] for idx in idxs])


def read_data():
    """
    Read data and return X, y pairs.
    """
    data = pickle.load(open('data/methods_tokens_graphs2.pkl', 'rb'))
    methods_source = data['methods_source']
    methods_graphs = data['methods_graphs']
    methods_names = data['methods_names']

    pairs = [((methods_source[i], methods_graphs[i]), methods_names[i]) for i in range(len(
        methods_source))]
    np.random.shuffle(pairs)

    return pairs


def read_tokens():
    """
        Read data and return X, y pairs without graph information.
    """
    data = pickle.load(open('data/methods_tokens_graphs2.pkl', 'rb'))
    methods_source = data['methods_source']
    methods_names = data['methods_names']

    pairs = [(methods_source[i], methods_names[i]) for i in range(len(methods_source))]
    np.random.shuffle(pairs)

    return pairs


def prepare_tokens(num_samples=None):
    """
        Prepare data and return language and X, y pairs.
    """
    lang = TokenLang('code')
    pairs = read_tokens()
    pairs = pairs if not num_samples else pairs[:num_samples]
    print("Read %s sentence pairs" % len(pairs))
    for pair in pairs:
        lang.add_sentence(pair[0])
        lang.add_sentence(pair[1])
    print("Counted words:")
    print(lang.name, lang.n_words)
    return lang, pairs


def prepare_data(num_samples=None):
    """
        Prepare data and return language and X, y pairs without graph information.
    """
    lang = TokenLang('code')
    pairs = read_data()
    pairs = pairs if not num_samples else pairs[:num_samples]
    print("Read %s sentence pairs" % len(pairs))
    for pair in pairs:
        lang.add_sentence(pair[0][0])
        lang.add_sentence(pair[1])
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


def sparse_adj_from_edges(edges):
    """
    Return a sparse Pytorch matrix given a list of edges.
    """
    f = [e[0] for e in edges]
    t = [e[1] for e in edges]
    n_nodes = max(f + t) + 1
    idxs = torch.LongTensor(edges)
    values = torch.ones(len(edges))

    adj = torch.sparse.FloatTensor(idxs.t(), values, torch.Size([n_nodes, n_nodes]))
    return adj


def tensors_from_pair_tokens_graph(pair, lang):
    """
    Get tensor from training given a X, y pair.
    """
    input_tensor = tensor_from_sentence_tokens(lang, pair[0][0])
    input_adj = sparse_adj_from_edges(pair[0][1][0])
    node_features = torch.tensor(pair[0][1][1])
    target_tensor = tensor_from_sentence_tokens(lang, pair[1])
    return (input_tensor, input_adj, node_features), target_tensor


def plot_loss(train_losses, val_losses, file_path='plots/loss.jpg'):
    """
    Plot the train and validation loss.
    """
    plt.clf()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(('train loss', 'validation loss'), loc='upper right')
    plt.title('Losses during training of LSTM->LSTM Model')
    plt.xlabel('#epochs')
    plt.ylabel('cross-entropy loss')
    pylab.savefig(file_path)
