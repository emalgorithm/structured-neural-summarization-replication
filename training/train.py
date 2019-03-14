import os
import argparse

from models.full_model import FullModel
from training.train_model import train_iters
from models.lstm_encoder import LSTMEncoder
from models.lstm_decoder import LSTMDecoder
from data_processing.data_util import prepare_tokens, prepare_data
from models.gat_encoder import GATEncoder
from models.gcn_encoder import GCNEncoder


def main():
    """
    Entry-point for running the models.
    """

    # Create directory for saving results
    model_dir = '../results/{}/'.format(opt.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Store hyperparams
    with open(model_dir + 'hyperparams.txt', 'w') as f:
        f.write(str(opt))

    # Prepare data
    if opt.graph:
        lang, pairs = prepare_data(num_samples=opt.n_samples)
        pairs = [pair for pair in pairs if len(pair[0][1][0]) > 0]
    else:
        lang, pairs = prepare_tokens(num_samples=opt.n_samples)

    # Create model
    hidden_size = 256
    encoder = LSTMEncoder(lang.n_words, hidden_size, opt.device).to(opt.device)

    decoder = LSTMDecoder(hidden_size, lang.n_words, opt.device, attention=opt.attention).to(
        opt.device)
    if opt.graph:
        if opt.gat:
            graph_encoder = GATEncoder(hidden_size, hidden_size)
        else:
            graph_encoder = GCNEncoder(hidden_size, hidden_size)
        model = FullModel(encoder=encoder, graph_encoder=graph_encoder, decoder=decoder,
                          device=opt.device)
    else:
        model = FullModel(encoder=encoder, decoder=decoder, device=opt.device)

    # Train model
    train_iters(model, opt.iterations, pairs, print_every=opt.print_every, model_dir=model_dir,
                lang=lang, graph=opt.graph)


parser = argparse.ArgumentParser()
parser.add_argument('--attention', type=bool, default=False, help='whether to use attention')
parser.add_argument('--model_name', default="test10", help='model name')
parser.add_argument('--device', default="cpu", help='cpu or cuda')
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to train on')
parser.add_argument('--print_every', type=int, default=1000, help='Print results after a fixed '
                                                                  'number of iterations')
parser.add_argument('--iterations', type=int, default=100, help='Number of iterations to train for')
parser.add_argument('--graph', type=bool, default=False, help='Whether to use a graph encoder')
parser.add_argument('--gat', type=bool, default=False, help='Whether to use GAT or GCN')

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":
    main()
