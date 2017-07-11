"""Replication of the model presented in "Learning Natural Language Inference
using Bidirectional LSTM model and Inner-Attention"
https://arxiv.org/pdf/1605.09090.pdf
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

from repeval.layers import (WordRepresentationLayer,
                            ContextRepresentationLayer,
                            MeanPoolingLayer,
                            LinearAggregationLayer,
                            GatherLastLayer,
                            DenseLayer)


class AttentionLayer(nn.Module):
    """Word-level attention as seen in https://arxiv.org/pdf/1605.09090.pdf

    Linear part inspired on
    http://pytorch.org/docs/_modules/torch/nn/modules/linear.html#Linear
    """
    def __init__(self, input_size, out_size):
        """input_size should be hidden_x_dirs"""
        super(AttentionLayer, self).__init__()
        self.input_size = input_size
        # self.W is W^y in the paper
        self.W = Parameter(torch.Tensor(self.input_size, self.input_size))
        self.W_h = Parameter(torch.Tensor(self.input_size, self.input_size))
        # self.context is w in the paper
        self.context = Parameter(torch.Tensor(self.input_size))
        self.reset_parameters()

    def forward(self, sent_batch, mean_sent_batch, batch_mask):
        """
        dim(sent_batch) = (seq_len, batch_size, hidden_x_dirs)
        dim(mean_sent_batch) = (batch_size, hidden_x_dirs)
        dim(batch_mask) = (seq_len, batch_size)
        dim(self.W) = (hidden_x_dirs, hidden_x_dirs)
        """
        seq_len = sent_batch.size(0)
        batch_size = sent_batch.size(1)
        hidden_x_dirs = sent_batch.size(2)

        assert batch_size == mean_sent_batch.size(0)
        assert self.input_size == hidden_x_dirs
        assert hidden_x_dirs == mean_sent_batch.size(1)

        # we need dim(sent_batch) = (batch_size, seq_len * hidden_x_dirs)
        # so after this op we'll have
        # dim(out) = (batch_size, seq_len * hidden_x_dirs)

        # -> (batch_size, hidden_x_dirs, hidden_x_dirs)
        W = self.W.unsqueeze(0).expand(batch_size,
                                       hidden_x_dirs,
                                       hidden_x_dirs)
        W_h = self.W_h.unsqueeze(0).expand(batch_size,
                                           hidden_x_dirs,
                                           hidden_x_dirs)

        # -> (seq_len, batch_size, hidden_x_dirs)
        batch_mask = batch_mask.unsqueeze(2).expand_as(sent_batch)
        # Make padding values = 0
        r_sent_batch = torch.mul(batch_mask, sent_batch)

        # -> (batch_size, seq_len, hidden_x_dirs)
        r_sent_batch = r_sent_batch.transpose(0, 1).contiguous()

        # expand mean_sent_batch
        # -> (batch_size, 1, hidden_x_dirs)
        mean_sent_batch = mean_sent_batch.unsqueeze(1)

        # -> (batch_size, seq_len, hidden_x_dirs)
        # Expanding instead of repeating throws an error
        # mean_sent_batch = mean_sent_batch.expand(batch_size,
        #                                          seq_len,
        #                                          hidden_x_dirs)
        mean_sent_batch = mean_sent_batch.repeat(1, seq_len, 1)

        WY = torch.bmm(r_sent_batch, W)
        WR = torch.bmm(mean_sent_batch, W_h)
        # dim(M) = (batch_size, seq_len, hidden_x_dirs)
        M = F.tanh(torch.add(WY, WR))

        # -> (batch_size, hidden_x_dirs)
        context = self.context.unsqueeze(0).expand(batch_size, hidden_x_dirs)
        # -> (batch_size, hidden_x_dirs, 1)
        context = context.unsqueeze(2)

        # dim(alpha) = (batch_size, seq_len, 1)
        alpha = F.softmax(torch.bmm(M, context))
        # -> (batch_size, 1, seq_len)
        alpha = alpha.transpose(1, 2)

        # -> (batch_size, 1, hidden_x_dirs)
        out = torch.bmm(alpha, r_sent_batch)
        # -> (batch_size, hidden_x_dirs)
        out = out.squeeze()

        return out

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.W_h.data.uniform_(-stdv, stdv)
        self.context.data.uniform_(-stdv, stdv)


class InnerAtt(nn.Module):
    def __init__(self, embeddings, num_classes, args, char_embeddings=None,
                 pos_embeddings=None):
        super(InnerAtt, self).__init__()
        self.embeddings = embeddings
        self.embedding_dim = embeddings.embedding_dim
        self.char_embeddings = char_embeddings
        self.char_hidden_dim = 50
        self.pos_embeddings = pos_embeddings
        self.num_classes = num_classes
        self.lstm_hidden_size = args.lstm_hidden
        self.lstm_layers = args.lstm_layers
        self.args = args
        self.dropout = args.dropout
        self.bidirectional = True
        self.num_dirs = 2 if self.bidirectional else 1
        self.hidden_x_dirs = self.lstm_hidden_size * self.num_dirs
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=True)
        self.use_cuda = args.cuda

        # self.sentence_input_layer = SentenceInput(embeddings, trainable=False)
        self.word_repr_layer = WordRepresentationLayer(
                          embeddings,
                          char_embeddings=self.char_embeddings,
                          char_hidden_dim=self.char_hidden_dim,
                          pos_embeddings=self.pos_embeddings,
                          char_merging_method=args.char_merging_method,
                          train_char_embeddings=args.train_char_embeddings,
                          cuda=self.use_cuda)
        self.context_embedding_dim = self.word_repr_layer.embedding_dim

        self.sentence_encoding_layer = ContextRepresentationLayer(
                                     embedding_dim=self.context_embedding_dim,
                                     n_layers=self.lstm_layers,
                                     hidden_size=self.lstm_hidden_size,
                                     bidirectional=self.bidirectional,
                                     dropout=self.dropout,
                                     separate_lstms=False,
                                     cuda=self.use_cuda)

        self.mean_pooling_layer = MeanPoolingLayer()
        # self.gather_last_layer = GatherLastLayer()
        self.attention_layer_p = AttentionLayer(self.hidden_x_dirs,
                                                self.hidden_x_dirs)
        self.attention_layer_h = AttentionLayer(self.hidden_x_dirs,
                                                self.hidden_x_dirs)
        self.aggregation_layer = LinearAggregationLayer()
        # 4 because we have prems, multiplication, difference and hyps tensors
        self.dense_layer = DenseLayer(4 * self.num_dirs *
                                      self.lstm_hidden_size,
                                      self.num_classes, dropout=self.dropout)
        # import ipdb; ipdb.set_trace()

    def forward(self, premises_tuple, hypotheses_tuple,
                char_premises_tuple=None, char_hypotheses_tuple=None,
                pos_premises_tuple=None, pos_hypotheses_tuple=None):
        premises_batch = premises_tuple[0]
        hypotheses_batch = hypotheses_tuple[0]

        premises_lengths = Variable(torch.LongTensor(premises_tuple[1]),
                                    requires_grad=False)
        hypotheses_lengths = Variable(torch.LongTensor(hypotheses_tuple[1]),
                                      requires_grad=False)
        if self.use_cuda:
            premises_lengths = premises_lengths.cuda()
            hypotheses_lengths = hypotheses_lengths.cuda()

        p_mask = premises_tuple[2]
        h_mask = hypotheses_tuple[2]
        # emb_[]_batch dim is (max_seq_len, batch_size, embedding_dim)
        # emb_prem_batch, emb_hypo_batch = self.sentence_input_layer(
        #                                                     premises_batch,
        #                                                     hypotheses_batch)

        char_premises_batch = None
        char_hypotheses_batch = None
        char_prem_masks = None
        char_hypo_masks = None
        char_prem_word_lens = None
        char_hypo_word_lens = None

        pos_premises_batch = None
        pos_hypotheses_batch = None

        if char_premises_tuple and char_hypotheses_tuple:
            # char_[]_tuple = (batch, sent_lengths, masks, word_lengths)
            char_premises_batch = char_premises_tuple[0]
            char_prem_masks = char_premises_tuple[2]
            char_prem_word_lens = char_premises_tuple[3]

            char_hypotheses_batch = char_hypotheses_tuple[0]
            char_hypo_masks = char_hypotheses_tuple[2]
            char_hypo_word_lens = char_hypotheses_tuple[3]

        if pos_premises_tuple and pos_hypotheses_tuple:
            pos_premises_batch = pos_premises_tuple[0]
            pos_hypotheses_batch = pos_hypotheses_tuple[0]

        # emb_[]_batch dim is (max_seq_len, batch_size, embedding_dim)
        (emb_prem_batch,
         emb_hypo_batch) = self.word_repr_layer(
                              premises_batch,
                              hypotheses_batch,
                              char_prem_batch=char_premises_batch,
                              char_hypo_batch=char_hypotheses_batch,
                              char_prem_masks=char_prem_masks,
                              char_hypo_masks=char_hypo_masks,
                              prem_word_lengths=char_prem_word_lens,
                              hypo_word_lengths=char_hypo_word_lens,
                              train_word_embeddings=self.args.train_embeddings,
                              pos_prem_batch=pos_premises_batch,
                              pos_hypo_batch=pos_hypotheses_batch)

        # We make sure the batch sizes are the same
        prem_batch_size = emb_prem_batch.size(1)
        hypo_batch_size = emb_hypo_batch.size(1)
        assert prem_batch_size == hypo_batch_size

        context_zero_state = self.sentence_encoding_layer.zero_state(
                                                              prem_batch_size)

        # dim([]_output) = (seq_len, batch_size, hidden_size*num_dirs)
        prem_output, hypo_output = self.sentence_encoding_layer(
                                                        emb_prem_batch,
                                                        emb_hypo_batch,
                                                        context_zero_state)

        prem_repr, hypo_repr = self.mean_pooling_layer(prem_output,
                                                       hypo_output,
                                                       premises_lengths,
                                                       hypotheses_lengths,
                                                       p_mask,
                                                       h_mask)

        prem_repr = self.attention_layer_p(prem_output, prem_repr, p_mask)
        hypo_repr = self.attention_layer_h(hypo_output, hypo_repr, h_mask)

        combined_vec = self.aggregation_layer(prem_repr, hypo_repr)
        class_activations = self.dense_layer(combined_vec)

        return class_activations, prem_repr, hypo_repr
