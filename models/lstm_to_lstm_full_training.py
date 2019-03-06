from __future__ import unicode_literals, print_function, division
import random
from tokens_util import tensors_from_pair_tokens, plot_loss, tensors_from_pair_tokens_graph

import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import f1_score
import numpy as np
from metrics import compute_rouge_scores
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(seq2seq_model, eval_pairs, criterion, eval='val', graph=False):
    with torch.no_grad():
        loss = 0
        f1 = 0
        rouge_2 = 0
        rouge_l = 0
        for i in range(len(eval_pairs)):
            if graph:
                eval_pair = eval_pairs[i]
                input_tensor = eval_pair[0][0].to(device)
                adj_tensor = eval_pair[0][1].to(device)
                node_features = eval_pair[0][2].to(device)
                target_tensor = eval_pair[1].to(device)

                output = seq2seq_model(sequence=input_tensor.view(-1), adj=adj_tensor,
                                       target=target_tensor.view(-1), node_features=node_features)
            else:
                eval_pair = eval_pairs[i]
                input_tensor = eval_pair[0]
                target_tensor = eval_pair[1]

                output = seq2seq_model(sequence=input_tensor.view(-1), target=target_tensor.view(
                    -1))

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


def train(input_tensor, target_tensor, seq2seq_model, optimizer, criterion, graph,
          adj_tensor=None, node_features=None):
    optimizer.zero_grad()

    if graph:
        output = seq2seq_model(sequence=input_tensor.view(-1), adj=adj_tensor,
                               target=target_tensor.view(-1), node_features=node_features)
    else:
        output = seq2seq_model(sequence=input_tensor.view(-1), target=target_tensor.view(-1))

    loss = criterion(output.view(-1, output.size(2)), target_tensor.view(-1))
    pred = output.view(-1, output.size(2)).argmax(1).cpu().numpy()

    loss.backward()

    optimizer.step()

    return loss.item(), pred


def train_iters(seq2seq_model, n_iters, pairs, print_every=1000, learning_rate=0.001,
                model_dir=None, lang=None, graph=False):
    train_losses = []
    val_losses = []

    val_f1_scores = []
    val_rouge_2_scores = []
    val_rouge_l_scores = []

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    f1 = 0
    rouge_2 = 0
    rouge_l = 0

    train_pairs, val_pairs, test_pairs = np.split(pairs,
                                                  [int(.8 * len(pairs)), int(.9 * len(pairs))])

    optimizer = optim.Adam(seq2seq_model.parameters(), lr=learning_rate)

    if graph:
        training_pairs = [tensors_from_pair_tokens_graph(random.choice(train_pairs), lang)
                          for i in range(n_iters)]
        val_tensor_pairs = [tensors_from_pair_tokens_graph(val_pair, lang) for val_pair in val_pairs]
    else:
        training_pairs = [tensors_from_pair_tokens(random.choice(train_pairs), lang)
                          for i in range(n_iters)]
        val_tensor_pairs = [tensors_from_pair_tokens(val_pair, lang) for val_pair in val_pairs]

    # test_tensor_pairs = [tensors_from_pair_tokens(test_pair, lang) for test_pair in test_pairs]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        if graph:
            input_tensor = training_pair[0][0].to(device)
            adj_tensor = training_pair[0][1].to(device)
            node_features = training_pair[0][2].to(device)
            target_tensor = training_pair[1].to(device)

            loss, pred = train(input_tensor, target_tensor, seq2seq_model, optimizer,
                               criterion, adj_tensor=adj_tensor, graph=graph, node_features=node_features)
        else:
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss, pred = train(input_tensor, target_tensor, seq2seq_model, optimizer, criterion,
                               graph=graph)

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
                                                          criterion, graph=graph)
            # test_loss, test_f1, test_rouge_2, test_rouge_l = evaluate(seq2seq_model,
            #                                                          test_tensor_pairs,
            #                                                       criterion, eval='test')

            if not val_losses or val_loss < min(val_losses):
                torch.save(seq2seq_model.state_dict(), model_dir + 'model.pt')
                print("Saved updated model")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # test_losses.append(test_loss)

            val_f1_scores.append(val_f1)
            val_rouge_2_scores.append(val_rouge_2)
            val_rouge_l_scores.append(val_rouge_l)

            pickle.dump([train_losses, val_losses, val_f1_scores, val_rouge_2_scores,
                         val_rouge_l_scores],
                        open('results/res.pkl', 'wb'))

            plot_loss(train_losses, val_losses, file_path=model_dir + 'loss.jpg')
