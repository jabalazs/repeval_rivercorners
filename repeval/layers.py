import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F
import torch.nn.init as weight_init
from repeval.utils.ops import embed_context_window


class LSTMCharEncoder(nn.Module):
    def __init__(self, char_embeddings, hidden_size, bidirectional=True,
                 train_char_embeddings=False, cuda=True):

        super(LSTMCharEncoder, self).__init__()
        self.char_embeddings = char_embeddings
        self.char_embedding_dim = self.char_embeddings.embedding_dim
        self.train_char_embeddings = train_char_embeddings
        self.n_layers = 1
        self.num_dirs = 2 if bidirectional else 1
        self.char_hidden_dim = hidden_size
        self.use_cuda = cuda
        self.char_lstm = nn.LSTM(self.char_embedding_dim,
                                 self.char_hidden_dim,
                                 self.n_layers,
                                 bidirectional=False,
                                 dropout=0.0)

    def forward(self, char_prem_batch, char_hypo_batch,
                char_prem_masks=None, char_hypo_masks=None,
                prem_word_lengths=None, hypo_word_lengths=None,
                zero_state=None):

        prem_seq_len = char_prem_batch.size(0)
        prem_word_len = char_prem_batch.size(1)
        batch_size = char_prem_batch.size(2)

        hypo_seq_len = char_hypo_batch.size(0)
        hypo_word_len = char_hypo_batch.size(1)
        assert batch_size == char_hypo_batch.size(2)

        if not self.train_char_embeddings:
            # Make embeddings not trainable:
            char_prem_batch = Variable(char_prem_batch.data,
                                       requires_grad=False)
            char_hypo_batch = Variable(char_hypo_batch.data,
                                       requires_grad=False)

        char_prem_batch = char_prem_batch.view(
                                    prem_seq_len * prem_word_len,
                                    batch_size)

        char_hypo_batch = char_hypo_batch.view(
                                    hypo_seq_len * hypo_word_len,
                                    batch_size)

        # -> ([]_seq_len * []_word_len, batch_size, embedding_dim)
        emb_char_prem_batch = self.char_embeddings(char_prem_batch)
        emb_char_hypo_batch = self.char_embeddings(char_hypo_batch)

        # dim([]_out)
        #  -> ([]_seq_len * []_word_len, batch_size, char_hidden_dim)
        prem_out, prem_hidden_tuple = self.char_lstm(emb_char_prem_batch,
                                                     zero_state)
        hypo_out, hypo_hidden_tuple = self.char_lstm(emb_char_hypo_batch,
                                                     zero_state)

        # -> (prem_seq_len, prem_word_len, batch_size, char_hidden_dim)
        prem_out = prem_out.contiguous().view(prem_seq_len,
                                              prem_word_len,
                                              batch_size,
                                              self.char_hidden_dim)
        hypo_out = hypo_out.contiguous().view(hypo_seq_len,
                                              hypo_word_len,
                                              batch_size,
                                              self.char_hidden_dim)

        # -> (prem_seq_len, 1, batch_size, 1)
        prem_word_lengths = prem_word_lengths.view(prem_seq_len, 1,
                                                   batch_size, 1)

        hypo_word_lengths = hypo_word_lengths.view(hypo_seq_len, 1,
                                                   batch_size, 1)

        # -> (prem_seq_len, 1, batch_size, char_hidden_dim)
        prem_word_lengths = prem_word_lengths.repeat(1, 1, 1,
                                                     self.char_hidden_dim)
        hypo_word_lengths = hypo_word_lengths.repeat(1, 1, 1,
                                                     self.char_hidden_dim)

        # -> (prem_seq_len, 1, batch_size, char_hidden_dim)
        new_prem_out = torch.gather(prem_out, 1, prem_word_lengths-1)
        new_hypo_out = torch.gather(hypo_out, 1, hypo_word_lengths-1)

        # -> (prem_seq_len, batch_size, char_hidden_dim)
        new_prem_out = new_prem_out.squeeze(1)
        new_hypo_out = new_hypo_out.squeeze(1)

        return new_prem_out, new_hypo_out

    def zero_state(self, batch_size=1):
        h_0 = Variable(torch.zeros(self.n_layers * self.num_dirs,
                                   batch_size,
                                   self.char_hidden_dim))

        c_0 = Variable(torch.zeros(self.n_layers * self.num_dirs,
                                   batch_size,
                                   self.char_hidden_dim))

        if self.use_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        return (h_0, c_0)


class LinearCharEncoder(nn.Module):

    def __init__(self, char_embeddings, train_char_embeddings=False,
                 char_merging_method='mean'):
        super(LinearCharEncoder, self).__init__()
        self.char_embeddings = char_embeddings
        self.char_embedding_dim = self.char_embeddings.embedding_dim
        self.train_char_embeddings = train_char_embeddings
        self.char_merging_method = char_merging_method

    def forward(self, char_prem_batch, char_hypo_batch,
                char_prem_masks=None, char_hypo_masks=None,
                prem_word_lengths=None, hypo_word_lengths=None,
                zero_state=None):
        """dim(char_prem_batch): (seq_len, word_len, batch_size)
           dim(char_prem_masks): (seq_len, word_len, batch_size)"""

        prem_seq_len = char_prem_batch.size(0)
        prem_word_len = char_prem_batch.size(1)
        batch_size = char_prem_batch.size(2)

        hypo_seq_len = char_hypo_batch.size(0)
        hypo_word_len = char_hypo_batch.size(1)
        assert batch_size == char_hypo_batch.size(2)

        if not self.train_char_embeddings:
            # Make embeddings not trainable:
            char_prem_batch = Variable(char_prem_batch.data,
                                       requires_grad=False)
            char_hypo_batch = Variable(char_hypo_batch.data,
                                       requires_grad=False)

        # -> ([]_seq_len * []_word_len, batch_size)
        char_prem_batch = char_prem_batch.view(
                                    prem_seq_len * prem_word_len,
                                    batch_size)

        char_hypo_batch = char_hypo_batch.view(
                                    hypo_seq_len * hypo_word_len,
                                    batch_size)

        # -> ([]_seq_len * []_word_len, batch_size, embedding_dim)
        emb_char_prem_batch = self.char_embeddings(char_prem_batch)
        emb_char_hypo_batch = self.char_embeddings(char_hypo_batch)

        # -> ([]_seq_len * []_word_len, batch_size, 1)
        char_prem_masks = char_prem_masks.view(prem_seq_len * prem_word_len,
                                               batch_size).unsqueeze(2)
        char_hypo_masks = char_hypo_masks.view(hypo_seq_len * hypo_word_len,
                                               batch_size).unsqueeze(2)

        # -> ([]_seq_len * []_word_len, batch_size, embedding_dim)
        char_prem_masks = char_prem_masks.repeat(1, 1, self.char_embedding_dim)
        char_hypo_masks = char_hypo_masks.repeat(1, 1, self.char_embedding_dim)

        masked_prem_batch = torch.mul(emb_char_prem_batch, char_prem_masks)
        masked_hypo_batch = torch.mul(emb_char_hypo_batch, char_hypo_masks)

        masked_prem_batch = masked_prem_batch.contiguous()
        masked_hypo_batch = masked_hypo_batch.contiguous()

        # -> ([]_seq_len, []_word_len, batch_size, embedding_dim)
        masked_prem_batch = masked_prem_batch.view(prem_seq_len,
                                                   prem_word_len,
                                                   batch_size,
                                                   self.char_embedding_dim)

        masked_hypo_batch = masked_hypo_batch.view(hypo_seq_len,
                                                   hypo_word_len,
                                                   batch_size,
                                                   self.char_embedding_dim)

        if self.char_merging_method == 'mean':
            agg_fn = torch.mean
        elif self.char_merging_method == 'sum':
            agg_fn = torch.sum

        # -> (prem_seq_len, batch_size, char_hidden_dim)
        agg_prem_batch = agg_fn(masked_prem_batch, 1).squeeze(1)
        agg_hypo_batch = agg_fn(masked_hypo_batch, 1).squeeze(1)

        return agg_prem_batch, agg_hypo_batch

    def __repr__(self):
        s = '{name}('
        s += 'train_char_embeddings={train_char_embeddings}'
        s += ', char_merging_method={char_merging_method}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class WordRepresentationLayer(nn.Module):

    def __init__(self, embeddings, char_embeddings=None,
                 pos_embeddings=None, **kwargs):
        """
        Transform tensor of word ids into vectors exctracted from the embedding
        lookup table.

        Args:
            embeddings: torch-initialized embeddings"""
        super(WordRepresentationLayer, self).__init__()
        self.embeddings = embeddings
        self.embedding_dim = embeddings.embedding_dim
        self.char_embeddings = char_embeddings
        self.train_char_embeddings = kwargs.get('train_char_embeddings',
                                                False)
        self.use_cuda = kwargs.get('cuda', True)

        if self.char_embeddings:
            self.char_merging_method = kwargs.get('char_merging_method', 'sum')
            char_hidden_dim = kwargs.get('char_hidden_dim', 50)
            bidirectional = kwargs.get('bidirectional', False)

            if self.char_merging_method == 'lstm':
                self.char_encoder = LSTMCharEncoder(
                              char_embeddings,
                              char_hidden_dim,
                              bidirectional,
                              train_char_embeddings=self.train_char_embeddings,
                              cuda=self.use_cuda)

                self.embedding_dim += char_hidden_dim

            elif self.char_merging_method in ['mean', 'sum']:
                self.char_encoder = LinearCharEncoder(
                              char_embeddings,
                              train_char_embeddings=self.train_char_embeddings,
                              char_merging_method=self.char_merging_method)

                self.embedding_dim += self.char_embeddings.embedding_dim
            else:
                raise NotImplementedError

        self.pos_embeddings = pos_embeddings
        if self.pos_embeddings:
            self.embedding_dim += self.pos_embeddings.embedding_dim

    def forward(self, premises_batch, hypotheses_batch,
                char_prem_batch=None, char_hypo_batch=None,
                char_prem_masks=None, char_hypo_masks=None,
                pos_prem_batch=None, pos_hypo_batch=None,
                prem_word_lengths=None, hypo_word_lengths=None,
                train_word_embeddings=False, invert_premises=False):
        """
        Args:
            premises_batch: (seq_len, batch_size)
            hypotheses_batch: IDEM

            prem_word_lengths: a padded tensor of word lengths, with dim
                (prem_seq_len, batch_size)
            hypo_word_lengths: a padded tensor of word lengths, with dim
                (hypo_seq_len, batch_size)

        return:
            emb_prem_batch: (seq_len, batch_size, embedding_dim)
            emb_hypo_batch: IDEM
        """

        prem_batch_dim = len(premises_batch.size())
        hypo_batch_dim = len(hypotheses_batch.size())
        batch_size = premises_batch.size(1)

        if prem_batch_dim == 3 and hypo_batch_dim == 3:
            emb_prem_batch = embed_context_window(self.embeddings,
                                                  premises_batch)
            emb_hypo_batch = embed_context_window(self.embeddings,
                                                  hypotheses_batch)
            if not train_word_embeddings:
                # Make embeddings not trainable:
                emb_prem_batch = Variable(emb_prem_batch.data,
                                          requires_grad=False)
                emb_hypo_batch = Variable(emb_hypo_batch.data,
                                          requires_grad=False)

            return (emb_prem_batch, emb_hypo_batch)

        elif prem_batch_dim == 2 and hypo_batch_dim == 2:
            emb_prem_batch = self.embeddings(premises_batch)
            emb_hypo_batch = self.embeddings(hypotheses_batch)

        if not train_word_embeddings:
            # Make embeddings not trainable:
            emb_prem_batch = Variable(emb_prem_batch.data, requires_grad=False)
            emb_hypo_batch = Variable(emb_hypo_batch.data, requires_grad=False)

        if char_prem_batch and char_hypo_batch:
            # dim(char_[]_batch) = (seq_len, word_len, batch_size)
            assert self.char_embeddings, ('You are feeding char_ids, but '
                                          'provided no char embeddings when '
                                          'initializing the layer')

            zero_state = None
            if self.char_merging_method == 'lstm':
                zero_state = self.char_encoder.zero_state(batch_size)

            new_prem_out, new_hypo_out = self.char_encoder(
                                           char_prem_batch,
                                           char_hypo_batch,
                                           char_prem_masks=char_prem_masks,
                                           char_hypo_masks=char_hypo_masks,
                                           prem_word_lengths=prem_word_lengths,
                                           hypo_word_lengths=hypo_word_lengths,
                                           zero_state=zero_state)

            emb_prem_batch = torch.cat([emb_prem_batch, new_prem_out], 2)
            emb_hypo_batch = torch.cat([emb_hypo_batch, new_hypo_out], 2)

        if pos_prem_batch and pos_hypo_batch:
            assert self.pos_embeddings, ('You are feeding pos_ids, but '
                                         'provided no pos embeddings when '
                                         'initializing the layer')

            emb_pos_prem_batch = self.pos_embeddings(pos_prem_batch)
            emb_pos_hypo_batch = self.pos_embeddings(pos_hypo_batch)

            emb_prem_batch = torch.cat([emb_prem_batch, emb_pos_prem_batch], 2)
            emb_hypo_batch = torch.cat([emb_hypo_batch, emb_pos_hypo_batch], 2)

        if invert_premises:

            inv_dim = 0  # dimension to invert
            inv_idx = torch.range(emb_prem_batch.size(inv_dim)-1, 0, -1).long()
            inv_idx = Variable(inv_idx)
            if self.use_cuda:
                inv_idx = inv_idx.cuda()
            emb_prem_batch = emb_prem_batch.index_select(inv_dim, inv_idx)

        return (emb_prem_batch, emb_hypo_batch)


class ContextRepresentationLayer(nn.Module):
    """RNN for encoding context"""
    def __init__(self, embedding_dim, n_layers, hidden_size, bidirectional,
                 dropout, separate_lstms=False, cuda=True):
        super(ContextRepresentationLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_dirs = 1 if not self.bidirectional else 2
        self.dropout = dropout
        self.separate_lstms = separate_lstms
        self.use_cuda = cuda

        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_size,
                            self.n_layers,
                            bidirectional=self.bidirectional,
                            dropout=self.dropout)

        if self.separate_lstms:
            self.lstm_hypo = nn.LSTM(self.embedding_dim,
                                     self.hidden_size,
                                     self.n_layers,
                                     bidirectional=self.bidirectional,
                                     dropout=self.dropout)

    def forward(self, emb_premises_batch, emb_hypotheses_batch, zero_state):
        # import ipdb; ipdb.set_trace()
        prem_out, prem_hidden_tuple = self.lstm(emb_premises_batch,
                                                zero_state)
        if self.separate_lstms:
            hypo_out, hypo_hidden_tuple = self.lstm_hypo(emb_hypotheses_batch,
                                                         zero_state)
        else:
            hypo_out, hypo_hidden_tuple = self.lstm(emb_hypotheses_batch,
                                                    zero_state)

        return(prem_out, hypo_out)

    def zero_state(self, batch_size=1):
        # both h_0 and c_0 must have dim
        # (num_layers * num_dirs, batch_size, hidden_size)
        h_0 = Variable(torch.zeros(self.n_layers * self.num_dirs,
                                   batch_size,
                                   self.hidden_size))

        c_0 = Variable(torch.zeros(self.n_layers * self.num_dirs,
                                   batch_size,
                                   self.hidden_size))

        if self.use_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        return (h_0, c_0)


class SelfAttentiveLayer(nn.Module):

    def __init__(self, input_size, d, r):
        """
        Taken from the paper
            "A structured self-attentive sentence embedding" ICLR 2017
            https://openreview.net/pdf?id=BJC_jUqxe

        :param d: size of the first atention layer
        :param r: number of attention "perspectives"
        """
        super(SelfAttentiveLayer, self).__init__()
        self.input_size = input_size
        self.d = d
        self.r = r
        self.W_s1 = Parameter(torch.Tensor(self.d, self.input_size))
        self.W_s2 = Parameter(torch.Tensor(self.r, self.d))

        self.reset_parameters()

    def forward(self, sent_batch, batch_mask):
        """

        :param sent_batch:
            Size is (seq_len, , batch_size, hidden_x_dirs)

        :param batch_mask:
            Size is (input_len, input_batch_size)

        :return:
            M: Size is (batch_size, self.r, hidden_x_dirs)

            A: Size is (batch_size, self.r, seq_len)

        """

        seq_len = sent_batch.size(0)
        batch_size = sent_batch.size(1)
        hidden_x_dirs = sent_batch.size(2)
        assert self.input_size == hidden_x_dirs

        # -> (batch_size, self.d, self.input_size)
        W_s1 = self.W_s1.unsqueeze(0).expand(batch_size,
                                             self.d,
                                             self.input_size)

        # -> (batch_size, self.r, self.d)
        W_s2 = self.W_s2.unsqueeze(0).expand(batch_size,
                                             self.r,
                                             self.d)

        # -> (seq_len, batch_size, hidden_x_dirs)
        batch_mask = batch_mask.unsqueeze(2).expand_as(sent_batch)

        # Make padding values = 0
        H = torch.mul(batch_mask, sent_batch)

        # -> (batch_size, seq_len, hidden_x_dirs)
        H = H.transpose(0, 1).contiguous()

        # -> (batch_size, hidden_x_dirs, seq_len)
        H_t = H.transpose(1, 2).contiguous()

        # -> (batch_size, self.d, seq_len)
        W_s1H_t = torch.bmm(W_s1, H_t)

        # -> (batch_size, self.d, seq_len)
        tanhW_s1H_t = F.tanh(W_s1H_t)

        # -> (batch_size, self.r, seq_len)
        W_s2tanhW_s1H_t = torch.bmm(W_s2, tanhW_s1H_t)

        # we need seq_len as first dimension because
        # torch softmax always uses this dimension to normalize
        # -> (seq_len, self.r, batch_size)
        W_s2tanhW_s1H_t = W_s2tanhW_s1H_t.transpose(0, 2)

        # -> (seq_len, self.r, batch_size)
        A = F.softmax(W_s2tanhW_s1H_t)

        # -> (batch_size, self.r, seq_len)
        A = A.transpose(0, 2)

        # -> (batch_size, self.r, hidden_x_dirs)
        M = torch.bmm(A, H)

        return M, A

    def reset_parameters(self):

        tanh_gain = weight_init.calculate_gain('tanh')
        linear_gain = weight_init.calculate_gain('linear')

        weight_init.xavier_uniform(self.W_s1.data, tanh_gain)
        weight_init.xavier_uniform(self.W_s2.data, linear_gain)


class QueryAttentionLayer(nn.Module):

    def __init__(self, input_size, hidden_size,
                 query_hidden_size=0, num_perspectives=1):
        """

        A more general version of SelfAttentive layer,
        allowing the usage of a "query" which is concatenated
        to the input when computing similarities.

        Based on the paper:
            "Grammar as a Foreign Language" NIPS 2015
            http://papers.nips.cc/paper/5635-grammar-as-a-foreign-language


        :param d: size of the first atention layer
        :param r: number of attention "perspectives"
        """
        super(QueryAttentionLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_perspectives = num_perspectives
        self.query_hidden_size = query_hidden_size
        if query_hidden_size:
            self.input_size += query_hidden_size
        self.W_s1 = Parameter(torch.Tensor(self.input_size,
                                           self.hidden_size))

        self.W_s2 = Parameter(torch.Tensor(self.hidden_size,
                                           self.num_perspectives))

    def forward(self, input_batch, input_batch_mask,
                query_batch=None, query_batch_mask=None):
        """
        :param input_batch:
            Size is (input_len, input_batch_size, input_hidden_size)
        :param input_batch_mask:
            Size is (input_len, input_batch_size)
        :param query_batch:
            Size is (query_len, query_batch_size, query_hidden_size)
        :param query_batch_mask:
            Size is (query_len, query_batch_size)
        :return:
        - M: Matrix of transformed inputs
            Size is (batch_size, query_len, self.num_perspectives,
                     input_hidden_size)
        - A: Matrix of attention values
            Size is (batch_size, input_len, query_len, self.num_perspectives)
        """

        input_len = input_batch.size(0)
        input_batch_size = input_batch.size(1)
        input_hidden_size = input_batch.size(2)
        query_len = 1
        query_hidden_size = 0

        # -> (input_batch_size, self.hidden_size, self.input_size)
        W_s1 = self.W_s1.unsqueeze(0).expand(input_batch_size,
                                             self.input_size,
                                             self.hidden_size)

        # -> (input_batch_size, self.num_perspectives, self.hidden_size)
        W_s2 = self.W_s2.unsqueeze(0).expand(input_batch_size,
                                             self.hidden_size,
                                             self.num_perspectives)

        # -> (input_len, input_batch_size, input_hidden_size)
        input_batch_mask = input_batch_mask.unsqueeze(2).expand_as(input_batch)

        # Make padding values = 0 in input_batch
        input_batch = torch.mul(input_batch, input_batch_mask)

        # -> (input_batch_size, input_len, input_hidden_size)
        input_batch = input_batch.transpose(0, 1).contiguous()

        # -> (input_batch_size, input_len, 1, input_hidden_size)
        H = input_batch.unsqueeze(2)

        if query_batch:
            assert self.query_hidden_size
            query_len = query_batch.size(0)
            query_batch_size = query_batch.size(1)
            query_hidden_size = query_batch.size(2)

            assert self.query_hidden_size == query_hidden_size
            assert input_batch_size == query_batch_size

            # -> (input_len, input_batch_size, input_hidden_size)
            query_batch_mask = query_batch_mask.unsqueeze(2).expand_as(query_batch)

            # Make padding values = 0 in query_batch
            query_batch = torch.mul(query_batch, query_batch_mask)

            # -> (query_batch_size, query_len, query_hidden_size)
            query_batch = query_batch.transpose(0, 1).contiguous()

            # -> (query_batch_size, 1, query_len, query_hidden_size)
            query_batch = query_batch.unsqueeze(1)

            # -> (query_batch_size, input_len, query_len, query_hidden_size)
            query_batch = query_batch.repeat(1, input_len, 1, 1)

            # -> (input_batch_size, input_len, query_len, input_hidden_size)
            H = H.repeat(1, 1, query_len, 1)

            # -> (input_batch_size, input_len, query_len, input_hidden_size + query_hidden_size )
            H = torch.cat([H, query_batch], 3)

        assert input_hidden_size + query_hidden_size == self.input_size

        H = H.contiguous()

        # -> (input_batch_size, input_len*query_len, self.input_size)
        H = H.view(input_batch_size, input_len*query_len, self.input_size)

        # -> (input_batch_size, input_len*query_len, self.hidden_size)
        HW = F.tanh(torch.bmm(H, W_s1))

        # -> (input_batch_size, input_len*query_len, self.num_perspectives)
        WHW = torch.bmm(HW, W_s2)

        # -> (input_batch_size, input_len, query_len, self.num_perspectives)
        WHW = WHW.view(input_batch_size, input_len, query_len, self.num_perspectives)

        # we need seq_len as second dimension because
        # torch softmax will use this dimension to normalize (we have a 4-tensor)

        # -> (input_batch_size, input_len, query_len, self.num_perspectives)
        A = F.softmax(WHW)

        # -> (input_batch_size, input_len, query_len*self.num_perspectives)
        A_t = A.view(input_batch_size, input_len, query_len*self.num_perspectives)

        # -> (input_batch_size, query_len*self.num_perspectives, input_len)
        A_t = A_t.transpose(1, 2)

        # -> (input_batch_size, query_len*self.num_perspectives, input_hidden_size)
        M = torch.bmm(A_t, input_batch)

        # -> (input_batch_size, query_len, self.num_perspectives, input_hidden_size)
        M = M.view(input_batch_size, query_len, self.num_perspectives, input_hidden_size)

        return M, A


class CoattentionLayer(nn.Module):

    def __init__(self):
        """

        Taken from the paper:
            "Dynamic coattention Networks for Question Answering" ICLR 2017
            https://arxiv.org/abs/1611.01604


        """
        super(CoattentionLayer, self).__init__()

    def forward(self, d_matrix_batch, q_matrix_batch):
        """

        :param d_matrix_batch:
            Size is (d_batch_size, d_input_size, hidden_size)

        :param q_matrix_batch:
            Size is (q_batch_size, q_input_size, hidden_size)


        Where input_sizes generally represent sequence lengths,
        and hidden sizes of the matrices have to match.

        :return:
            C_D, size is (batch_size,  d_input_size,2*hidden_size)

        """

        d_batch_size = d_matrix_batch.size(0)
        d_input_size = d_matrix_batch.size(1)
        d_hidden_size = d_matrix_batch.size(2)

        q_batch_size = q_matrix_batch.size(0)
        q_input_size = q_matrix_batch.size(1)
        q_hidden_size = q_matrix_batch.size(2)

        assert d_batch_size == q_batch_size
        assert d_hidden_size == q_hidden_size

        # -> (batch_size,  d_input_size, hidden_size)
        D_t = d_matrix_batch

        # -> (batch_size, hidden_size, q_input_size)
        Q = q_matrix_batch.transpose(1, 2).contiguous()

        # -> (batch_size, d_input_size, q_input_size)
        L = torch.bmm(D_t, Q)

        # FOR D

        # -> (batch_size, q_input_size, d_input_size)
        L_D = L.transpose(1, 2).contiguous()

        # -> (q_input_size, batch_size, d_input_size)
        L_D = L_D.transpose(0, 1).contiguous()

        # -> (q_input_size, batch_size, d_input_size)
        A_D = F.softmax(L_D)

        # -> (batch_size, q_input_size, d_input_size)
        A_D = A_D.transpose(0, 1).contiguous()

        # FOR Q

        # -> (d_input_size, batch_size, q_input_size)
        L_Q = L.transpose(0, 1).contiguous()

        # -> (d_input_size, batch_size, q_input_size)
        A_Q = F.softmax(L_Q)

        # -> (batch_size, d_input_size, q_input_size)
        A_Q = A_Q.transpose(0, 1).contiguous()

        # -> (batch_size,  hidden_size, d_input_size)
        D = D_t.transpose(1, 2)

        # -> (batch_size,  hidden_size, q_input_size)
        C_Q = torch.bmm(D, A_Q)

        # -> (batch_size,  2*hidden_size, q_input_size)
        merged = torch.cat([Q, C_Q], 1)

        # -> (batch_size,  2*hidden_size, d_input_size)
        C_D = torch.bmm(merged, A_D)

        # -> (batch_size,  d_input_size, 2*hidden_size)
        C_D = C_D.transpose(1, 2).contiguous()

        return C_D


class DenseLayer(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.0):
        super(DenseLayer, self).__init__()
        self.dropout = dropout

        self.linear_layer1 = nn.Linear(input_size, 2000)
        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=True)
        self.linear_layer2 = nn.Linear(2000, 2000)
        self.linear_layer3 = nn.Linear(2000, num_classes)

    def forward(self, input_batch):
        out1 = F.relu(self.linear_layer1(input_batch))
        self.dropout_layer(out1)
        out2 = F.relu(self.linear_layer2(out1))
        self.dropout_layer(out2)
        logits = self.linear_layer3(out2)
        return logits


class LinearAggregationLayer(nn.Module):

    def __init__(self):
        """

        Simply concatenate the provided tensors on their last dimension
        which needs to have the same size,  along with their
        element-wise multiplication and difference

        Taken from the paper:
             "Learning Natural Language Inference using Bidirectional
             LSTM model and Inner-Attention"
             https://arxiv.org/abs/1605.09090

        """
        super(LinearAggregationLayer, self).__init__()

    def forward(self, input_1, input_2):
        """

        :param : input_1
            Size is (*, hidden_size)

        :param input_2:
            Size is (*, hidden_size)

        :return:

            Merged vectors, size is (*, 4*hidden size)
        """
        assert input_1.size(-1) == input_2.size(-1)
        mult_combined_vec = torch.mul(input_1, input_2)
        diff_combined_vec = torch.abs(input_1 - input_2)
        combined_vec = torch.cat((input_1,
                                  input_2,
                                  mult_combined_vec,
                                  diff_combined_vec), input_1.dim()-1)

        return combined_vec


class GatedEncoderLayer(nn.Module):

    def __init__(self, I, J, K, F):

        """

        Taken from the appendix of the paper
            "A structured self-attentive sentence embedding" ICLR 2017
            https://openreview.net/pdf?id=BJC_jUqxe

        Inspired in:
        https://www.iro.umontreal.ca/~memisevr/pubs/pami_relational.pdf

        :param input_size:
        :param hidden_size:
        :param f_size: numer of factors
        """

        super(GatedEncoderLayer, self).__init__()
        self.I = I
        self.J = J
        self.K = K
        self.F = F

        self.W_x = Parameter(torch.Tensor(self.I, self.F))

        self.W_y = Parameter(torch.Tensor(self.J, self.F))

        self.W_z = Parameter(torch.Tensor(self.K, self.F))

        self.reset_parameters()



    def reset_parameters(self):

        linear_gain = weight_init.calculate_gain('linear')

        weight_init.xavier_uniform(self.W_x.data, linear_gain)
        weight_init.xavier_uniform(self.W_y.data, linear_gain)
        weight_init.xavier_uniform(self.W_z.data, linear_gain)

    def forward(self, matrix_batch):
        """

        :param matrix_batch:
            Expected size is (batch_size, input_size, hidden_size)

        :return:
            Factored input, size is (batch_size, input_size, self.f_size)
        """

        batch_size = matrix_batch.size(0)
        input_size = matrix_batch.size(1)
        hidden_size = matrix_batch.size(2)

        assert self.I == input_size
        assert self.J == hidden_size

        # -> (I, 1, 1, F)
        e_W_x = self.W_x.unsqueeze(1).unsqueeze(1)

        # -> (I, J, K, F)
        e_W_x = e_W_x.repeat(1, self.J, self.K, 1)

        # -> (1, J, 1, F)
        e_W_y = self.W_y.unsqueeze(1).unsqueeze(0)

        # -> (I, J, K, F)
        e_W_y = e_W_y.repeat(self.I, 1, self.K, 1)

        # -> (1, 1, K, F)
        e_W_z = self.W_z.unsqueeze(0).unsqueeze(0)

        # -> (I, J, K, F)
        e_W_z = self.W_z.repeat(self.I, self.J, 1, 1)

        W = torch.mul(e_W_x, e_W_y).mul(e_W_z).sum(3).squeeze()

        # -> (batch_size*input_size, hidden_size)
        matrix_batch = matrix_batch.contiguous().view(batch_size*input_size,
                                                      hidden_size)

        # -> (batch_size*input_size, 1, hidden_size)
        b_matrix_batch = matrix_batch.unsqueeze(1)

        # -> (1, input_size, hidden_size, self.K)
        b_w = W.unsqueeze(0)

        # -> (batch_size, input_size, hidden_size, self.K)
        b_w = b_w.repeat(batch_size, 1, 1, 1).contiguous()

        # -> (batch_size*input_size, hidden_size, self.f_size)
        b_w = b_w.view(batch_size*input_size, hidden_size, self.K)

        # -> (batch_size*input_size, 1, self.K)
        b_F = torch.bmm(b_matrix_batch, b_w)

        # -> (batch_size*input_size, self.K)
        b_F = b_F.squeeze().contiguous()

        # -> (batch_size, input_size, self.f_size)
        out = b_F.view(batch_size, input_size, self.K)

        return out


class MeanPoolingLayer(nn.Module):

    def __init__(self):
        super(MeanPoolingLayer, self).__init__()

    def forward(self, prem_out, hypo_out, prem_lengths, hypo_lengths,
                p_mask, h_mask):

        """

        :param prem_out:
            Size is  (prem_seq_len, batch_size, prem_hidden_dim)

        :param hypo_out:
            Size is (hypo_seq_len, batch_size, hypo_hidden_dim)

        :param prem_lengths:
            Size is (batch_size)

        :param hypo_lengths:
            Size is (batch_size)

        :param p_mask:
            Size is (prem_seq_len, batch_size)

        :param h_mask:
            Size is (hypo_seq_len, batch_size)

        :return:
            Tuple of prems and hypothesis averaged on their
            respective sequence lengths, skipping paddings.

            Sizes are (batch_size, prem_hidden_sim) and
            (batch_size, hypo_hidden_sim)

        """

        batch_size = prem_out.size(1)
        assert batch_size == hypo_out.size(1)

        hidden_x_dirs = prem_out.size(2)
        assert hidden_x_dirs == hypo_out.size(2)
        # the masks are of dimension (seq_len, batch_size) and we need them
        # to be (seq_len, batch_size, hidden_x_dirs)
        p_mask = p_mask.view(-1, batch_size, 1).repeat(1, 1, hidden_x_dirs)
        h_mask = h_mask.view(-1, batch_size, 1).repeat(1, 1, hidden_x_dirs)
        masked_prem = torch.mul(prem_out, p_mask)
        masked_hypo = torch.mul(hypo_out, h_mask)

        # now, we want to sum the tensors along the sequence dimension which
        # will result in tensors of dimension (batch_size, hidden_x_dirs)
        masked_prem = torch.sum(masked_prem, 0).squeeze()
        masked_hypo = torch.sum(masked_hypo, 0).squeeze()

        # for calculating the average we now broadcast the lengths:
        # (batch_size) -> (batch_size, hidden_x_dirs)
        prem_lengths = prem_lengths.view(batch_size, 1)
        prem_lengths = prem_lengths.repeat(1, hidden_x_dirs)
        hypo_lengths = hypo_lengths.view(batch_size, 1)
        hypo_lengths = hypo_lengths.repeat(1, hidden_x_dirs)

        # finally we obtain the averages
        # (batch_size, hidden_x_dirs)
        prem_lengths = prem_lengths.float()
        hypo_lengths = hypo_lengths.float()
        prem_sentence_vec = torch.div(masked_prem, prem_lengths)
        hypo_sentence_vec = torch.div(masked_hypo, hypo_lengths)

        return(prem_sentence_vec, hypo_sentence_vec)


class SumPoolingLayer(nn.Module):

    def __init__(self):
        super(SumPoolingLayer, self).__init__()

    def forward(self, prem_out, hypo_out, p_mask, h_mask):

        """
        :param prem_out:
            Size is  (prem_seq_len, batch_size, prem_hidden_dim)
        :param hypo_out:
            Size is (hypo_seq_len, batch_size, hypo_hidden_dim)
        :param p_mask:
            Size is (prem_seq_len, batch_size)
        :param h_mask:
            Size is (hypo_seq_len, batch_size)
        :return:
            Tuple of prems and hypothesis summed on their
            respective sequence lengths, skipping paddings.
            Sizes are (batch_size, prem_hidden_sim) and
            (batch_size, hypo_hidden_sim)
        """

        batch_size = prem_out.size(1)
        assert batch_size == hypo_out.size(1)

        hidden_x_dirs = prem_out.size(2)
        assert hidden_x_dirs == hypo_out.size(2)
        # the masks are of dimension (seq_len, batch_size) and we need them
        # to be (seq_len, batch_size, hidden_x_dirs)
        p_mask = p_mask.view(-1, batch_size, 1).repeat(1, 1, hidden_x_dirs)
        h_mask = h_mask.view(-1, batch_size, 1).repeat(1, 1, hidden_x_dirs)
        masked_prem = torch.mul(prem_out, p_mask)
        masked_hypo = torch.mul(hypo_out, h_mask)

        # now, we want to sum the tensors along the sequence dimension which
        # will result in tensors of dimension (batch_size, hidden_x_dirs)
        sum_prem = torch.sum(masked_prem, 0).squeeze()
        sum_hypo = torch.sum(masked_hypo, 0).squeeze()

        return (sum_prem, sum_hypo)


class MaxPoolingLayer(nn.Module):

    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def forward(self, prem_out, hypo_out, p_mask, h_mask):

        """
        :param prem_out:
            Size is  (prem_seq_len, batch_size, prem_hidden_dim)
        :param hypo_out:
            Size is (hypo_seq_len, batch_size, hypo_hidden_dim)
        :param prem_lengths:
            Size is (batch_size)
        :param hypo_lengths:
            Size is (batch_size)
        :param p_mask:
            Size is (prem_seq_len, batch_size)
        :param h_mask:
            Size is (hypo_seq_len, batch_size)
        :return:
            Tuple of prems and hypothesis averaged on their
            respective sequence lengths, skipping paddings.
            Sizes are (batch_size, prem_hidden_sim) and
            (batch_size, hypo_hidden_sim)
        """

        batch_size = prem_out.size(1)
        assert batch_size == hypo_out.size(1)

        hidden_x_dirs = prem_out.size(2)
        assert hidden_x_dirs == hypo_out.size(2)
        # the masks are of dimension (seq_len, batch_size) and we need them
        # to be (seq_len, batch_size, hidden_x_dirs)
        p_mask = p_mask.view(-1, batch_size, 1).repeat(1, 1, hidden_x_dirs)
        h_mask = h_mask.view(-1, batch_size, 1).repeat(1, 1, hidden_x_dirs)
        masked_prem = torch.mul(prem_out, p_mask)
        masked_hypo = torch.mul(hypo_out, h_mask)

        # now, we want to sum the tensors along the sequence dimension which
        # will result in tensors of dimension (batch_size, hidden_x_dirs)
        max_prem, _ = torch.max(masked_prem, 0)
        max_hypo, _ = torch.max(masked_hypo, 0)

        max_prem = max_prem.squeeze()
        max_hypo = max_hypo.squeeze()

        return (max_prem, max_hypo)


class GatherLastLayer(nn.Module):

    def __init__(self):
        """

        Return the last hidden state of a tensor returned by an RNN

        """
        super(GatherLastLayer, self).__init__()

    def forward(self, lstm_out, lengths):
        """

        Args:
            lstm_out: A Variable containing a 3D tensor of dimension
                (seq_len, batch_size, hidden_x_dirs)
            lengths: A Variable containing 1D LongTensor of dimension
                (batch_size)

        Return:
            A Variable containing a 2D tensor of the same type as lstm_out of
            dim (batch_size, hidden_x_dirs) corresponding to the concatenated
            last hidden states of the forward and backward parts of the input.
        """

        seq_len = lstm_out.size(0)
        batch_size = lstm_out.size(1)
        hidden_x_dirs = lstm_out.size(2)
        single_dir_hidden = hidden_x_dirs / 2

        lengths_fw = lengths
        lengths_bw = seq_len - lengths_fw

        rep_lengths_fw = lengths_fw.view(1, batch_size, 1)
        rep_lengths_fw = rep_lengths_fw.repeat(1, 1, single_dir_hidden)

        rep_lengths_bw = lengths_bw.view(1, batch_size, 1)
        rep_lengths_bw = rep_lengths_bw.repeat(1, 1, single_dir_hidden)

        # we want 2 chunks in the last dimension
        out_fw, out_bw = torch.chunk(lstm_out, 2, 2)

        h_t_fw = torch.gather(out_fw, 0, rep_lengths_fw-1)
        h_t_bw = torch.gather(out_bw, 0, rep_lengths_bw)

        # -> (batch_size, hidden_x_dirs)
        last_hidden_out = torch.cat([h_t_fw, h_t_bw], 2).squeeze()
        return last_hidden_out
