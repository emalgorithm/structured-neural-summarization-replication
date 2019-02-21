from __future__ import unicode_literals, print_function, division
import random
from models.lstm_encoder import LSTMEncoder
from models.lstm_decoder import LSTMDecoder
from models.gcn_encoder import GCNEncoder
from models.lstm_to_lstm import Seq2Seq
from models.seq_graph_to_seq import SeqGraph2Seq
from tokens_util import prepare_tokens, tensors_from_pair_tokens, plot_loss, prepare_data, tensors_from_pair_tokens_graph

import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import f1_score
import numpy as np
from metrics import compute_rouge_scores
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model running on {}".format(device))

lang, pairs = prepare_data()
pairs = [pair for pair in pairs if len(pair[0][1]) > 0]
# test_pairs = pairs[-10000:]
# val_pairs = pairs[-20000:-10000]
# train_pairs = pairs[:-20000]
# pairs = pairs[:100]
train_pairs, val_pairs, test_pairs = np.split(pairs, [int(.8*len(pairs)), int(.9*len(pairs))])

test_pairs = test_pairs
val_pairs = val_pairs
train_pairs = train_pairs


def evaluate(seq2seq_model, eval_pairs, criterion, eval='val'):
    with torch.no_grad():
        loss = 0
        f1 = 0
        rouge_2 = 0
        rouge_l = 0
        for i in range(len(eval_pairs)):
            eval_pair = eval_pairs[i]
            input_tensor = eval_pair[0][0].to(device)
            adj_tensor = eval_pair[0][1].to(device)
            target_tensor = eval_pair[1].to(device)

            output = seq2seq_model(input_tensor.view(-1), adj_tensor, target_tensor.view(-1))
            # output = seq2seq_model(input_tensor.view(-1), target_tensor.view(-1))
            loss += criterion(output.view(-1, output.size(2)), target_tensor.view(-1))
            pred = output.view(-1, output.size(2)).argmax(1).cpu().numpy()

            y_true = target_tensor.cpu().numpy().reshape(-1)
            f1 += f1_score(y_true, pred, average='micro')
            rouge_2_temp, rouge_l_temp = compute_rouge_scores(pred, y_true)
            rouge_2 += rouge_2_temp
            rouge_l += rouge_l_temp

        loss /= len(eval_pairs)
        f1 /= len(eval_pairs)
        rouge_2 /= len(eval_pairs)
        rouge_l /= len(eval_pairs)

        print('{} loss: {}'.format(eval, loss))
        print('{} f1_score: {}'.format(eval, f1))
        print('{} rouge_2_score: {}'.format(eval, rouge_2))
        print('{} rouge_l_score: {}'.format(eval, rouge_l))

        return loss, f1, rouge_2, rouge_l


def train(input_tensor, adj_tensor, target_tensor, seq2seq_model, optimizer, criterion):
    optimizer.zero_grad()

    output = seq2seq_model(input_tensor.view(-1), adj_tensor, target_tensor.view(-1))
    # output = seq2seq_model(input_tensor.view(-1), target_tensor.view(-1))
    loss = criterion(output.view(-1, output.size(2)), target_tensor.view(-1))
    pred = output.view(-1, output.size(2)).argmax(1).cpu().numpy()

    loss.backward()

    optimizer.step()

    return loss.item(), pred


def train_iters(seq2seq_model, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
    train_losses = []
    val_losses = []
    test_losses = []

    test_f1_scores = []
    test_rouge_2_scores = []
    test_rouge_l_scores = []

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    f1 = 0
    rouge_2 = 0
    rouge_l = 0

    optimizer = optim.Adam(seq2seq_model.parameters(), lr=learning_rate)

    # training_pairs = []
    # for i in range(n_iters):
    #     print(i)
    #     training_pairs.append(tensors_from_pair_tokens_graph(random.choice(train_pairs), lang))

    training_pairs = [tensors_from_pair_tokens_graph(random.choice(train_pairs), lang)
                      for i in range(n_iters)]



    val_tensor_pairs = [tensors_from_pair_tokens_graph(val_pair, lang) for val_pair in val_pairs]
    test_tensor_pairs = [tensors_from_pair_tokens_graph(test_pair, lang) for test_pair in test_pairs]
    criterion = nn.NLLLoss()

    print("Finished preparing data")

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0][0].to(device)
        adj_tensor = training_pair[0][1].to(device)
        target_tensor = training_pair[1].to(device)

        loss, pred = train(input_tensor, adj_tensor, target_tensor, seq2seq_model, optimizer,
                           criterion)
        print_loss_total += loss
        plot_loss_total += loss

        y_true = target_tensor.cpu().numpy().reshape(-1)

        if len(y_true) < len(pred):
            y_true = np.pad(y_true, (0, len(pred) - len(y_true)), mode='constant')
        else:
            pred = np.pad(pred, (0, len(y_true) - len(pred)), mode='constant')

        f1 += f1_score(y_true, pred, average='micro')
        rouge_2_temp, rouge_l_temp = compute_rouge_scores(pred, y_true)
        rouge_2 += rouge_2_temp
        rouge_l += rouge_l_temp

        # print("Pred: {}".format(lang.to_tokens(pred)))
        # print("Target: {}".format(lang.to_tokens(target_tensor.numpy().reshape(-1))))
        # print()

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('train (%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))
            print('train f1_score: {}'.format(f1 / iter))
            print('train rouge_2_score: {}'.format(rouge_2 / iter))
            print('train rouge_l_score: {}'.format(rouge_l / iter))

            train_loss = print_loss_avg
            val_loss, val_f1, val_rouge_2, val_rouge_l = evaluate(seq2seq_model, val_tensor_pairs,
                                                          criterion)
            test_loss, test_f1, test_rouge_2, test_rouge_l = evaluate(seq2seq_model,
                                                                     test_tensor_pairs,
                                                                  criterion, eval='test')

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)

            test_f1_scores.append(test_f1)
            test_rouge_2_scores.append(test_rouge_2)
            test_rouge_l_scores.append(test_rouge_l)

            pickle.dump([train_losses, val_losses, test_losses, test_f1_scores, test_rouge_2_scores,
                         test_rouge_l_scores],
                        open('results/res.pkl', 'wb'))

            plot_loss(train_losses, val_losses, test_losses)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0




hidden_size = 256
encoder1 = LSTMEncoder(lang.n_words, hidden_size).to(device)
graph_encoder = GCNEncoder(hidden_size, hidden_size)
attn_decoder1 = LSTMDecoder(hidden_size, lang.n_words).to(device)
lstm2lstm = SeqGraph2Seq(seq_encoder=encoder1, graph_encoder=graph_encoder,
                         seq_decoder=attn_decoder1, device=device)
# lstm2lstm = Seq2Seq(encoder=encoder1, decoder=attn_decoder1, device=device)
train_iters(lstm2lstm, 100000, print_every=100, plot_every=1000)