#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse

from repeval.utils.embeddings import (Glove840B300d, SennaEmbeddings)

from repeval.corpus.snli_corpus import (MultiNLICorpus,
                                        SNLICorpus,
                                        FullNLICorpus)

Corpora = {"MultiNLICorpus": MultiNLICorpus,
           "SNLICorpus": SNLICorpus,
           "FullNLICorpus": FullNLICorpus}

EmbeddingsList = {"Glove840B300d": Glove840B300d,
                  "SennaEmbeddings": SennaEmbeddings}

parser = argparse.ArgumentParser(description='PyTorch MultiNLI Classifier')

parser.add_argument('--corpus', type=str, default="MultiNLICorpus",
                    choices=Corpora,
                    help='Name of the corpus to use. '
                         'Choices: ' + " ".join(Corpora.keys()))

parser.add_argument('--embeddings', type=str, default="Glove840B300d",
                    choices=EmbeddingsList,
                    help='Name of the embeddings to use. '
                         'Choices: ' + " ".join(EmbeddingsList.keys()))

parser.add_argument('--context_window_size', type=int, default=1,
                    help='Size of the context window size')

parser.add_argument('--runs_dir', type=str,
                    default='./runs',
                    help='location of the runs directories')


parser.add_argument('--load', type=str,  default=None,
                    help='Model to load either for further training or eval')

parser.add_argument('--write_output', action='store_true',
                    help='Whether to write evaluation output to a file')

parser.add_argument('--dont_update_db', action='store_false', dest='update_db',
                    help='Deactivate database writing')

parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')

parser.add_argument('--train_embeddings', action='store_true',
                    help='Enable embedding training. Default: False')

parser.add_argument('--use_char_embeddings', action='store_true',
                    help='Enable character level embeddigs. Default: False')
parser.add_argument('--dont_train_char_embeddings', action='store_false',
                    dest='train_char_embeddings',
                    help='Enable character embedding training. Default: False')
parser.add_argument('--char_merging_method',
                    choices=['lstm', 'mean', 'sum'],
                    help='Way in which char embeddings are encoded to produce '
                         'word reprentations',
                    default='mean')
parser.add_argument('--char_embedding_dim', type=int, default=20,
                    help='dimension of the character embedding. Default 20')

parser.add_argument('--pos_tags', action='store_true',
                    help='Enable POS tag embeddings. Default: False')
parser.add_argument('--pos_embedding_dim', type=int, default=50,
                    help='dimension of the pos tags embeddings. Default 50')

parser.add_argument('--invert_premises', action='store_true',
                    help='Invert premises during training')
parser.add_argument('--double_hypotheses', action='store_true',
                    help='Double hypotheses during training')
parser.add_argument('--differentiate_inputs', action='store_true',
                    help='Delete common words between premise and hypotheses')

parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (0 = no dropout). '
                         'Default: 0.25')
parser.add_argument('--clip', type=float, default=5,
                    help='gradient clipping (default=5)')


# Learning Rate
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate Default 0.001')

parser.add_argument('--lr_decay', type=float, default=1,
                    help='learning rate decay (Default: 1, ie. no decay)')

parser.add_argument('--start_decay_at_epoch', type=float, default=1,
                    help='Epoch at which to begin decaying the learning rate. '
                         'Default: 1')


parser.add_argument('--update_learning_rate', action='store_true',
                    help='Do not decay learning rate (may be desirable for '
                         'some optimzers (e.g. Adam)')

parser.add_argument('--param_init', type=float, default=None,
                    help='Uniform distribution interval limit for '
                         'initializing model parameters. '
                         '[-param_init, param_init]. If None, torch'
                         'iitialization is used. Default: None')

parser.add_argument('--optim', default='adam',
                    choices=['sgd', 'adagrad', 'adadelta', 'adam', 'rmsprop'],
                    help='Optimization method. Default: adam')

parser.add_argument('--lstm_hidden', type=int, default=300,
                    help='Hidden dimension of the lstm. Default: 300')

parser.add_argument('--lstm_layers', type=int, default=1,
                    help='Number of layers for the lstm. Default: 1')

parser.add_argument('--epochs', type=int, default=10,
                    help='Upper epoch limit. Default: 10')

parser.add_argument('--early_stopping_epochs', type=int, default=None,
                    help="If validation accuracy hasn't improved in "
                         "early_stopping_epochs, training will stop. "
                         "Default: None (early stopping will not be applied)")

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size. Default: 128')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--no_cuda', action='store_false', dest='cuda',
                    help="Don't use CUDA")

parser.add_argument('--mode', type=str,  default='train',
                    choices=('train', 'test'),
                    help='Whether to train or eval')


def run_main():
    import torch.nn as nn
    from repeval.models.inner_att import InnerAtt
    from repeval.routines import main

    args = parser.parse_args()

    Model = InnerAtt

    loss_function = nn.CrossEntropyLoss()

    main(args, Model, Corpora, EmbeddingsList, loss_function)


if __name__ == "__main__":
    run_main()
