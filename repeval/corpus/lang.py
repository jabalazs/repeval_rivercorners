import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import repeval.constants as constants
from repeval.utils.io import save_pickle, load_pickle


class Lang:
    def __init__(self, corpus):
        self.name = corpus.name
        self.train_filepath = corpus.filepath
        self.word2index = {constants.PAD_TOKEN: constants.PAD_ID,
                           constants.UNK_TOKEN: constants.UNK_ID,
                           constants.SOS_TOKEN: constants.SOS_ID,
                           constants.EOS_TOKEN: constants.EOS_ID,
                           constants.NUM_TOKEN: constants.NUM_ID}
        self.index2word = {v: k for k, v in self.word2index.iteritems()}
        self.word2count = {constants.NUM_TOKEN: 0}
        self.n_words = 5  # Count PAD, UNK, SOS, EOS and NUM

        self.char2index = {constants.UNK_CHAR_TOKEN: constants.UNK_CHAR_ID}
        self.index2char = {v: k for k, v in self.char2index.iteritems()}
        self.char2count = {}
        self.n_chars = 1  # count UNK_CHAR

        self.pos2index = {constants.PAD_TOKEN: constants.PAD_ID,
                          constants.UNK_TOKEN: constants.UNK_ID,
                          constants.SOS_TOKEN: constants.SOS_ID,
                          constants.EOS_TOKEN: constants.EOS_ID,
                          constants.NUM_TOKEN: constants.NUM_ID}

        self.has_pos_tags = False
        self.index2pos = {v: k for k, v in self.pos2index.iteritems()}
        self.pos2count = {}
        self.n_pos = 1  # count UNK_POS

    def _get_high_frequency_words(self, low_freq_threshold):
        """Create dict containing words with a frequency greater than
           low_freq_threshold. This function should be called after the lang
           is complete"""
        if low_freq_threshold == 0:
            self.word2index_hf = self.word2index
            self.index2word_hf = self.index2word
            return

        self.word2index_hf = {constants.PAD_TOKEN: constants.PAD_ID,
                              constants.UNK_TOKEN: constants.UNK_ID,
                              constants.SOS_TOKEN: constants.SOS_ID,
                              constants.EOS_TOKEN: constants.EOS_ID,
                              constants.NUM_TOKEN: constants.NUM_ID}
        i = 5
        for word, freq in self.word2count.iteritems():
            if freq > low_freq_threshold:
                self.word2index_hf[word] = i
                i += 1
        self.index2word_hf = {v: k for k, v in self.word2index_hf.iteritems()}

    def read_add_sentence(self, sentence):
        """

        :param sentence:  CorpusSentence object
        :return:
        """
        for word in sentence.tokens:
            self.read_add_word(word)

        if sentence.pos_tags:
            if not self.has_pos_tags:
                self.has_pos_tags = True
            for pos_tag in sentence.pos_tags:
                self.read_add_pos_tag(pos_tag)

    def read_add_word(self, word):
        word = word.lower()
        num_exp = re.compile(r'[\$+-]?\d+(?:\.\d+)?')
        is_num = num_exp.match(word)
        if is_num:
            word = constants.NUM_TOKEN

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

        if is_num:
            return

        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.char2count[char] = 1
                self.index2char[self.n_chars] = char
                self.n_chars += 1
            else:
                self.char2count[char] += 1

    def read_add_pos_tag(self, pos_tag):
        if pos_tag not in self.pos2index:
            self.pos2index[pos_tag] = self.n_pos
            self.pos2count[pos_tag] = 1
            self.index2pos[self.n_pos] = pos_tag
            self.n_pos += 1
        else:
            self.pos2count[pos_tag] += 1

    def sentence2index(self, sentence):
        indexes = []
        for token in sentence.tokens:
            token = token.lower()
            indexes.append(self.word2index_hf.get(token, constants.UNK_ID))
        return indexes

    def sentence2posindex(self, sentence):
        indexes = []
        for pos_tag in sentence.pos_tags:
            try:
                indexes.append(self.pos2index[pos_tag])
            except KeyError:
                indexes.append(constants.UNK_ID)
        return indexes

    def wordidx2charids(self, wordidx):
        """Return a list of character indices from a single word index"""
        word = self.index2word_hf[wordidx]
        chars_idx = []
        for char in word:
            try:
                chars_idx.append(self.char2index[char])
            except KeyError:
                chars_idx.append(constants.UNK_CHAR_ID)
        return chars_idx

    def idsentence2charids(self, id_sentence):
        """Return a list of lists of char indices from a list of word indices.

        input: a list of word indices
        output: a list of lists of character indices"""

        char_indices = []
        for word_id in id_sentence:
            char_indices.append(self.wordidx2charids(word_id))
        return char_indices

    def get_raw_embeddings(self, embeddings_object):
        return embeddings_object()

    def get_torch_char_embeddings(self, dim_size=20, Embeddings=None):
        """Produce character embeddings
           args:
                dim_size: size of the embedding dimesion if initialize_from
                    arg is None
                Embeddings: embedding object from which to initialize the
                    character embeddigs"""
        root_path, ext = os.path.splitext(self.train_filepath)
        if Embeddings:
            pickle_path = (root_path +
                           '_{}_embeddings.pickle'.format(Embeddings.__name__))
        else:
            pickle_path = (root_path + '_char_embeddings_'
                           'd{}.pickle'.format(str(dim_size)))

        if os.path.exists(pickle_path):
            print('Loading embeddings {}'.format(pickle_path))
            return load_pickle(pickle_path)
        else:
            print('Creating torch char'
                  ' embeddings pickle: {}'.format(pickle_path))
            char_vocab_size = self.n_chars
            if Embeddings is None:
                embedding_matrix = np.random.uniform(-0.05,
                                                     0.05,
                                                     size=[char_vocab_size,
                                                           dim_size])
            else:
                embedding_matrix = self._load_char_embeddings(Embeddings)

            torch_embs = nn.Embedding(*embedding_matrix.shape)
            torch_embs.weight = Parameter(torch.FloatTensor(embedding_matrix))
            save_pickle(pickle_path, torch_embs)
            return torch_embs

    def get_torch_pos_embeddings(self, dim_size):
        root_path, ext = os.path.splitext(self.train_filepath)
        pickle_path = (root_path + '_pos_embeddings_'
                                   'd{}.pickle'.format(str(dim_size)))
        if os.path.exists(pickle_path):
            print('Loading embeddings {}'.format(pickle_path))
            return load_pickle(pickle_path)
        else:
            print('Creating torch pos'
                  ' embeddings pickle: {}'.format(pickle_path))
            pos_vocab_size = self.n_pos
            embedding_matrix = np.random.uniform(-0.05, 0.05,
                                                 size=[pos_vocab_size,
                                                       dim_size])
            torch_embs = nn.Embedding(*embedding_matrix.shape)
            torch_embs.weight = Parameter(torch.FloatTensor(embedding_matrix))
            save_pickle(pickle_path, torch_embs)
            return torch_embs

    def _load_embeddings(self, embeddings_object):
        """Create numpy embedding matrix from `embeddings`

            Args:
                embeddings: an Embeddings object created by Edison's code"""

        print('Loading {} for {}...'.format(embeddings_object.__name__,
                                            self.name))
        embeddings = self.get_raw_embeddings(embeddings_object)
        vocab_size = len(self.word2index_hf)
        vector_size = embeddings.vector_size
        embedding_matrix = np.zeros((vocab_size, vector_size),
                                    dtype=np.float32)

        # set the the unseen/unknown token embedding
        embedding_matrix[self.word2index_hf[constants.UNK_TOKEN]] = embeddings.unseen()

        for token, idx in self.word2index_hf.items():
            if token in embeddings:
                embedding_matrix[idx] = embeddings[token]
            else:
                embedding_matrix[idx] = embeddings.unseen()

        return embedding_matrix

    def _load_char_embeddings(self, embeddings_object):
        """Create numpy embedding matrix from `embeddings`

            Args:
                embeddings: an Embeddings object created by Edison's code"""

        print('Loading {} for {}...'.format(embeddings_object.__name__,
                                            self.name))
        embeddings = self.get_raw_embeddings(embeddings_object)
        char_vocab_size = self.n_chars
        vector_size = embeddings.vector_size
        embedding_matrix = np.zeros((char_vocab_size, vector_size),
                                    dtype=np.float32)

        unk_char_index = self.char2index[constants.UNK_CHAR_TOKEN]
        # set the the unseen/unknown token embedding
        embedding_matrix[unk_char_index] = embeddings.unseen()

        for token, idx in self.char2index.items():
            if token in embeddings:
                embedding_matrix[idx] = embeddings[token]
            else:
                embedding_matrix[idx] = embeddings.unseen()

        return embedding_matrix

    def get_torch_embeddings(self, embeddings):
        """Create torch Embedding object `embeddings`

            Args:
                embeddings: an Embeddings object created by Edison's code"""
        root_path, ext = os.path.splitext(self.train_filepath)
        pickle_path = (root_path +
                       '_{}_embeddings.pickle'.format(embeddings.__name__))
        if os.path.exists(pickle_path):
            print('Loading embeddings {}'.format(pickle_path))
            return load_pickle(pickle_path)
        else:
            print('Creating torch embeddings pickle: {}'.format(pickle_path))
            emb_matrix = self._load_embeddings(embeddings)
            all_embedding = nn.Embedding(*emb_matrix.shape,
                                         padding_idx=constants.PAD_ID)
            all_embedding.weight = Parameter(torch.FloatTensor(emb_matrix))
            save_pickle(pickle_path, all_embedding)
            return all_embedding
