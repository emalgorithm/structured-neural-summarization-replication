import os
import torch
import argparse

from models.lstm_to_lstm import Seq2Seq
from models.lstm_to_lstm_full_training import train_iters
from models.lstm_encoder import LSTMEncoder
from models.lstm_decoder import LSTMDecoder
from tokens_util import prepare_tokens, prepare_data
from models.gat_encoder import GATEncoder
from models.gcn_encoder import GCNEncoder


def main(model_name):
    model_dir = '../results/{}/'.format(model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if opt.graph:
        lang, pairs = prepare_data(num_samples=opt.n_samples)
        pairs = [pair for pair in pairs if len(pair[0][1]) > 0]
    else:
        lang, pairs = prepare_tokens(num_samples=opt.n_samples)

    hidden_size = 256
    encoder = LSTMEncoder(lang.n_words, hidden_size, opt.device).to(opt.device)

    decoder = LSTMDecoder(hidden_size, lang.n_words, opt.device, attention=opt.attention).to(
        opt.device)
    if opt.graph:
        if opt.gat:
            graph_encoder = GATEncoder(hidden_size, hidden_size)
        else:
            graph_encoder = GCNEncoder(hidden_size, hidden_size)
        model = Seq2Seq(encoder=encoder, graph_encoder=graph_encoder, decoder=decoder,
                        device=opt.device)
    else:
        model = Seq2Seq(encoder=encoder, decoder=decoder, device=opt.device)

    train_iters(model, opt.iterations, pairs, print_every=opt.print_every, model_dir=model_dir,
                lang=lang, graph=opt.graph)


parser = argparse.ArgumentParser()
parser.add_argument('--attention', type=bool, default=False, help='whether to use attention')
parser.add_argument('--model_name', default="test10", help='model name')
parser.add_argument('--device', default="cpu", help='cpu or cuda')
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to train on')
parser.add_argument('--print_every', type=int, default=1000, help='Number of samples to train on')
parser.add_argument('--iterations', type=int, default=100, help='Number of samples to train on')
parser.add_argument('--graph', type=bool, default=False, help='Number of samples to train on')
parser.add_argument('--gat', type=bool, default=False, help='Number of samples to train on')

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":
    main(opt.model_name)
