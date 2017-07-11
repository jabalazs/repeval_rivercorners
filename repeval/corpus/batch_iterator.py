import random
import torch
from torch.autograd import Variable
import repeval.constants as constants


def context_window(l, win, pad_id=constants.PAD_ID):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >= 1
    l = list(l)

    lpadded = win // 2 * [pad_id] + l + win // 2 * [pad_id]
    out = [lpadded[i:i + win] for i in range(len(l))]

    assert len(out) == len(l)

    return out


class BatchIterator(object):

    def __init__(self, id_tuples, batch_size, char_id_tuples=None,
                 pos_id_tuples=None,  shuffle=False, align_right=False,
                 cuda=False, context_window_size=1, allowed_label_ids=None,
                 double_hypotheses=False, differentiate_inputs=False,
                 max_prem_len=200):

        """Create batches of length batch_size from the id_tuples
        Args:
            id_tuples: Ad-hoc data type for SNLI data. Iterable containing
                3-tuples of [premise, hypothesis, label] where premise and
                hypothesis are lists of ints and label is an int.
            batch_size: The desired batch size.
            char_id_tuples: Optional argument. Iterable containing
                3-tuples of [premise, hypothesis, label] where premise and
                hypothesis are a list of lists of ints representing a
                character-level encoded sentence, and label is an int.
            shuffle: whether to shuffle the data before creating the batches.
            align_right: whether to align "right" or "left" within the batch.
                A left aligned batch will have the padding tokens before the
                valid tokens.
            cuda: whether to the return the batches as torch variables in the
                gpu or cpu.

        return the tuple:
            ((premises, premises_len, p_mask),
             (hypotheses, hyp_len, h_mask),
             labels)
            with premises_len and hyp_len included.
            premises and hypotheses are batches"""

        self.premises = []
        self.hypotheses = []
        self.char_premises = []
        self.char_hypotheses = []
        self.pos_premises = []
        self.pos_hypotheses = []
        self.labels = []
        self.ids = []
        self.align_right = align_right
        self.context_window_size = context_window_size
        self.allowed_label_ids = allowed_label_ids

        self.id_tuples = id_tuples
        self.char_id_tuples = char_id_tuples
        self.pos_id_tuples = pos_id_tuples

        tuples = [self.id_tuples]
        if char_id_tuples:
            tuples.append(self.char_id_tuples)
        if pos_id_tuples:
            tuples.append(self.pos_id_tuples)

        self.tuples = zip(*tuples)

        if shuffle:
            random.shuffle(self.tuples)

        for tuple in self.tuples:
            id_tuple = tuple[0]

            premise = id_tuple[0]
            if max_prem_len and len(premise) > max_prem_len:
                continue

            if double_hypotheses:
                hypothesis = id_tuple[1] + id_tuple[1]
            else:
                hypothesis = id_tuple[1]
            label = id_tuple[2]
            sent_id = id_tuple[3]

            if differentiate_inputs:
                prem_inter_hypo = set(premise).intersection(set(hypothesis))
                premise = [token if token not in prem_inter_hypo
                           else constants.UNK_ID for token in premise]

                hypothesis = [token if token not in prem_inter_hypo
                              else constants.UNK_ID for token in hypothesis]

            if self.allowed_label_ids and label not in self.allowed_label_ids:
                continue

            if self.context_window_size > 1:
                prem_context_win = context_window(premise,
                                                  self.context_window_size)
                hypo_context_win = context_window(hypothesis,
                                                  self.context_window_size)

                self.premises.append(torch.LongTensor(prem_context_win))
                self.hypotheses.append(torch.LongTensor(hypo_context_win))
            else:
                self.premises.append(torch.LongTensor(premise))
                self.hypotheses.append(torch.LongTensor(hypothesis))

            self.labels.append(label)
            self.ids.append(sent_id)

            if self.char_id_tuples:
                char_id_tuple = tuple[1]
                char_premise = char_id_tuple[0]
                char_hypothesis = char_id_tuple[1]
                self.char_premises.append([torch.LongTensor(char_premise_word)
                                           for char_premise_word
                                           in char_premise])
                self.char_hypotheses.append([torch.LongTensor(char_hypothesis_word)
                                             for char_hypothesis_word
                                             in char_hypothesis])

            if self.pos_id_tuples:
                if self.char_id_tuples:
                    pos_id_tuple = tuple[2]
                else:
                    pos_id_tuple = tuple[1]

                pos_premise = pos_id_tuple[0]
                pos_hypothesis = pos_id_tuple[1]

                self.pos_premises.append(torch.LongTensor(pos_premise))
                self.pos_hypotheses.append(torch.LongTensor(pos_hypothesis))

        self.labels = torch.LongTensor(self.labels)

        assert (len(self.premises) == len(self.hypotheses))
        self.cuda = cuda
        self.batch_size = batch_size
        self.num_batches = (len(self.premises) + batch_size - 1) // batch_size

        if self.cuda:
            self.labels = self.labels.cuda()

    def _pad_list(self, input_list, dim0_pad=None, dim1_pad=None,
                  pad_lengths=False):
        """Receive a list of lists and return a padded 2d torch tensor,
           a list of lengths and a padded mask

           input_list: a list of lists. len(input_list) = M, and N is the max
           length of any of the lists contained in input_list.

              e.g.: [[2,45,3,23,54], [12,4,2,2], [4], [45, 12]]

           Return a torch tensor of dimension (M, N) corresponding to the padded
           sequence, a list of the original lengths, and a mask
           Returns:
               out: a torch tensor of dimension (M, N)
               lengths: a list of ints containing the lengths of each input_list
                        element
               mask: a torch tensor of dimension (M, N)
           """
        if not dim0_pad:
            dim0_pad = len(input_list)

        if not dim1_pad:
            dim1_pad = max(x.size(0) for x in input_list)

        if self.context_window_size > 1:
            out = input_list[0].new(dim0_pad,
                                    dim1_pad,
                                    self.context_window_size)
            out = out.fill_(constants.PAD_ID)
        else:
            out = input_list[0].new(dim0_pad,
                                    dim1_pad).fill_(constants.PAD_ID)
        mask = torch.zeros(dim0_pad, dim1_pad)

        lengths = []
        for i in range(len(input_list)):
            data_length = input_list[i].size(0)
            ones = torch.ones(data_length)
            lengths.append(data_length)
            offset = dim1_pad - data_length if self.align_right else 0
            out[i].narrow(0, offset, data_length).copy_(input_list[i])
            mask[i].narrow(0, offset, data_length).copy_(ones)

        return out, lengths, mask

    def _pad_character_sentence(self, char_sent_batch):
        """char_sent_batch: A list containing character-encoded sentences.
           Each character-encoded sentence is a list of 1d tensors representing
           a character-encoded word.

           return (sent_batch, sent_lengths, word_lengths_tensor, masks) where
           sent_batch is a 3d tensor of dimension
           (max_sent_len, max_word_len, batch_size)
           sent_lengths is a 1d tensor of dim (batch_size)
           word_lengths_tensor is a 2d tensor of dim (max_sent_len, batch_size)
           """
        batch_size = len(char_sent_batch)
        # max word length for the whole batch
        max_word_len = max([max(len(word) for word in char_sent)
                           for char_sent in char_sent_batch])

        max_sent_len = max([len(char_sent) for char_sent in char_sent_batch])

        word_lengths_tensor = torch.ones(batch_size, max_sent_len).long()

        sent_batch = []
        sent_lengths = []
        masks = []
        for i, sentence in enumerate(char_sent_batch):
            padded_sent, word_lengths, mask = self._pad_list(sentence,
                                                             max_sent_len,
                                                             max_word_len)
            word_lengths = torch.LongTensor(word_lengths)
            word_lengths_tensor[i].narrow(0, 0, len(word_lengths)).copy_(word_lengths)

            if self.cuda:
                padded_sent = padded_sent.cuda()
                mask = mask.cuda()
                word_lengths_tensor = word_lengths_tensor.cuda()

            sent_batch.append(padded_sent.contiguous())
            sent_lengths.append(len(sentence))
            masks.append(mask.contiguous())

        word_lengths_tensor = Variable(word_lengths_tensor,
                                       requires_grad=False)
        word_lengths_tensor = word_lengths_tensor.transpose(0, 1).contiguous()

        sent_batch = torch.stack(sent_batch, dim=2)
        sent_batch = Variable(sent_batch, requires_grad=False)

        masks = torch.stack(masks, dim=2)
        masks = Variable(masks, requires_grad=False)

        sent_lengths = torch.LongTensor(sent_lengths)
        if self.cuda:
            sent_lengths = sent_lengths.cuda()
        sent_lengths = Variable(sent_lengths)

        return sent_batch, sent_lengths, masks, word_lengths_tensor

    def _batchify(self, data):
        out, lengths, mask = self._pad_list(data)
        out = out.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        if self.cuda:
            out = out.cuda()
            mask = mask.cuda()

        v = Variable(out, requires_grad=False)
        m = Variable(mask, requires_grad=False)
        return v, lengths, m

    def __getitem__(self, index):
        assert index < self.num_batches, ("Index is greater "
                                          "than the number of batches "
                                          "%d>%d" % (index, self.num_batches))

        # First we obtain the batch slices
        premises_slice = self.premises[index * self.batch_size:
                                       (index + 1) * self.batch_size]

        hypotheses_slice = self.hypotheses[index * self.batch_size:
                                           (index + 1) * self.batch_size]

        labels_batch = self.labels[index * self.batch_size:
                                   (index + 1) * self.batch_size]

        sent_ids_batch = self.ids[index * self.batch_size:
                                  (index + 1) * self.batch_size]

        # Then we pad and cast the padded batches into torch variables
        premises_batch, premises_lengths, p_mask = self._batchify(
                                                                premises_slice)

        hypotheses_batch, hypotheses_lengths, h_mask = self._batchify(
                                                              hypotheses_slice)

        result = [sent_ids_batch,
                  (premises_batch, premises_lengths, p_mask),
                  (hypotheses_batch, hypotheses_lengths, h_mask),
                  Variable(labels_batch, requires_grad=False)]

        # Now we repeat the same process for character-level encoded sentences
        if self.char_id_tuples:
            char_premises_slice = self.char_premises[index * self.batch_size:
                                                     (index + 1) * self.batch_size]

            char_hypotheses_slice = self.char_hypotheses[index * self.batch_size:
                                                         (index + 1) * self.batch_size]

            (char_prem_batch,
             prem_sent_lengths,
             char_prem_mask,
             prem_word_lengths) = self._pad_character_sentence(char_premises_slice)

            (char_hypo_batch,
             hypo_sent_lengths,
             char_hypo_mask,
             hypo_word_lengths) = self._pad_character_sentence(char_hypotheses_slice)

            result += [(char_prem_batch, prem_sent_lengths,
                        char_prem_mask, prem_word_lengths),
                       (char_hypo_batch, hypo_sent_lengths,
                        char_hypo_mask, hypo_word_lengths)]
        else:
            result += [None, None]

        # Now we repeat the same process for pos tags
        if self.pos_id_tuples:
            pos_premises_slice = \
                self.pos_premises[index * self.batch_size:(index + 1) * self.batch_size]

            pos_hypotheses_slice = \
                self.pos_hypotheses[index * self.batch_size:(index + 1) * self.batch_size]

            # Then we pad and cast the padded batches into torch variables
            # we ignore lengths and masks since they are equivalent to words
            pos_premises_batch, _, _ = self._batchify(pos_premises_slice)

            pos_hypotheses_batch, _, _ = self._batchify(pos_hypotheses_slice)

            result += [(pos_premises_batch,),
                       (pos_hypotheses_batch,)]

        else:
            result += [None, None]

        return result

    def __len__(self):
        return self.num_batches
