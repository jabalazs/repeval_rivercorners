import os
import gc
import json

import repeval.constants as constants
from repeval.corpus.tree import Tree
from repeval.corpus.lang import Lang
from repeval.corpus.batch_iterator import BatchIterator
from repeval.utils.io import save_pickle, load_pickle


CORPUS_PATH = os.path.join(constants.DATA_PATH, "corpus")
MULTINLI_CORPUS_PATH = os.path.join(CORPUS_PATH, "multinli_0.9")
SNLI_CORPUS_PATH = os.path.join(CORPUS_PATH, "snli_1.0")
FULLNLI_CORPUS_PATH = os.path.join(CORPUS_PATH, "fullnli")

TRAIN_SUFFIX = "_train.jsonl"
DEV_SUFFIX = "_dev.jsonl"
TEST_SUFFIX = "_test.jsonl"


def get_filename(path, mode):
    if mode not in ('train', 'dev', 'test'):
        raise Exception("Wrong dataset type")
    base_path, dataset_name = os.path.split(path)

    if mode == 'train':
        filename = dataset_name + TRAIN_SUFFIX

    elif mode == 'dev':
        filename = dataset_name + DEV_SUFFIX

    elif mode == 'test':
        filename = dataset_name + TEST_SUFFIX

    return os.path.join(path, filename)


class NLICorpusSentence(object):

    def __init__(self, string, tree_string, binary_tree_string, id=None):
        self.string = string
        self.tree_string = tree_string
        self.binary_tree_string = binary_tree_string
        tree = Tree.fromstring(tree_string)
        self.tokens, self.pos_tags = zip(*tree.pos())
        if id:
            self.id = id

    def __contains__(self, item):
        return item in self.tokens

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item]

    def __len__(self):
        return len(self.tokens)

    def count(self, value):
        return self.tokens.count(value)


class NLICorpusExample(object):

    def __init__(self, example):
        self.annotator_labels = example['annotator_labels']
        self.caption_id = example.get('captionID', None)
        self.pair_id = example.get('pairID', None)
        self.genre = example.get('genre', None)
        self.gold_label = example['gold_label']
        self.sentence_1 = NLICorpusSentence(example['sentence1'],
                                            example['sentence1_parse'],
                                            example['sentence1_binary_parse'])

        self.sentence_2 = NLICorpusSentence(example['sentence2'],
                                            example['sentence2_parse'],
                                            example['sentence2_binary_parse'])


class NLICorpus(object):
    """
    Class to read SNLI corpus.

    Note: The class construction is not optimized for efficiency but for easier
    understanding. We iterate several times over the data instead of doing
    everything in one pass.
    """

    path = None
    label_dict = None

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    @property
    def name(self):
        return str(self)

    @property
    def num_classes(self):
        return len(self.label_dict)

    def __init__(self, path=None, mode='train', from_raw=False, chars=False):
        """
        filepath: jsonl file path
        from_raw: whether to re load the corpus from raw example
                  otherwise, will just the read tuples from pre-procesed
                  pickle files
        """
        if path:
            self.path = path

        self.mode = mode
        self.chars = chars
        self.filepath = get_filename(self.path, self.mode)

        if self.mode == 'train':
            self.train_filepath = self.filepath
        else:
            self.train_filepath = get_filename(self.path, "train")

        self._path_root, self._ext = os.path.splitext(self.filepath)

        self.raw_examples = []
        self.tuples = []
        self.id_tuples = []
        self.from_raw = from_raw
        self.char_id_tuples = []
        self.pos_id_tuples = []

        if self.from_raw:
            self._load()

        self._create_tuples()
        self._create_lang()
        self.lang._get_high_frequency_words(low_freq_threshold=0)
        self._create_id_tuples()
        if self.chars:
            self._create_char_id_tuples()

        if self.lang.has_pos_tags:
            self._create_pos_id_tuples()

    def _load(self):
        pickle_path = self._path_root + '.pickle'
        if os.path.exists(pickle_path):
            print('Loading pickle: {}'.format(pickle_path))
            gc.disable()
            self.raw_examples = load_pickle(pickle_path)
            gc.enable()
        else:
            print('Reading files and creating {}'.format(pickle_path))
            self._read()
            save_pickle(pickle_path, self.raw_examples)

    def _read(self):
        with open(self.filepath, "r") as f:
            for line in f.readlines():
                example = json.loads(line)
                try:
                    if example['gold_label'] == '-':
                        print('Ignoring example with "-" label: {}'
                              ''.format(example["pairID"]))
                        continue
                    # Dirty hack for reading test set
                    elif example['gold_label'] == 'hidden':
                        example['gold_label'] = 'neutral'
                except KeyError:
                    pass
                self.raw_examples.append(NLICorpusExample(example))

    def _create_tuples(self):
        tuples_pickle_path = self._path_root + '_tuples.pickle'
        if os.path.exists(tuples_pickle_path):
            print('Loading tuples pickle: {}'.format(tuples_pickle_path))
            self.tuples = load_pickle(tuples_pickle_path)
        else:
            print("Creating tuples pickle {}".format(tuples_pickle_path))
            if not self.raw_examples:
                self._load()
            for idx, example in enumerate(self.raw_examples):
                premise = example.sentence_1
                hypothesis = example.sentence_2
                label_idx = self.label_dict[example.gold_label]
                try:
                    sent_id = example.pair_id
                except AttributeError:
                    sent_id = idx
                self.tuples.append((premise, hypothesis, label_idx, sent_id))

            save_pickle(tuples_pickle_path, self.tuples)

    def _create_lang(self):
        """Create the language belonging to the corpus

        For now we're going to hardcode the train corpus language
        as the universal one"""

        # TRAIN FILEPATH HARDCODED HERE
        # This code MUST be called first for the training set!!

        path_root, ext = os.path.splitext(self.train_filepath)
        lang_pickle_path = path_root + '_lang.pickle'
        if os.path.exists(lang_pickle_path):
            print('Loading Lang pickle: {}'.format(lang_pickle_path))
            self.lang = load_pickle(lang_pickle_path)
        else:
            print('Creating lang pickle {}'.format(lang_pickle_path))
            self.lang = Lang(self)
            for hypothesis, premise, label_idx, sent_id in self.tuples:
                self.lang.read_add_sentence(premise)
                self.lang.read_add_sentence(hypothesis)
            save_pickle(lang_pickle_path, self.lang)

    def _create_id_tuples(self):
        id_tuples_pickle_path = self._path_root + '_idtuples.pickle'

        if os.path.exists(id_tuples_pickle_path):

            print('Loading id tuples pickle {}'.format(id_tuples_pickle_path))
            self.id_tuples = load_pickle(id_tuples_pickle_path)

        else:
            print('Creating id tuples pickle {}'.format(id_tuples_pickle_path))
            for tupl in self.tuples:
                premise = tupl[0]
                hypothesis = tupl[1]
                label_id = tupl[2]
                sent_id = tupl[3]
                premise_ids = self.lang.sentence2index(premise)
                hypothesis_ids = self.lang.sentence2index(hypothesis)
                self.id_tuples.append((premise_ids, hypothesis_ids, label_id,
                                       sent_id))

            save_pickle(id_tuples_pickle_path, self.id_tuples)

    def _create_char_id_tuples(self):
        char_id_tuples_pickle_path = self._path_root + '_char_idtuples.pickle'

        if os.path.exists(char_id_tuples_pickle_path):

            print('Loading char id'
                  ' tuples pickle {}'.format(char_id_tuples_pickle_path))
            self.char_id_tuples = load_pickle(char_id_tuples_pickle_path)

        else:
            print('Creating char id tuples pickle '
                  '{}'.format(char_id_tuples_pickle_path))

            for tupl in self.id_tuples:
                premise_ids = tupl[0]
                hypothesis_ids = tupl[1]
                label_id = tupl[2]
                sent_id = tupl[3]

                premise_char_ids = self.lang.idsentence2charids(premise_ids)
                hypothesis_char_ids = self.lang.idsentence2charids(
                                                                hypothesis_ids)

                self.char_id_tuples.append((premise_char_ids,
                                            hypothesis_char_ids,
                                            label_id, sent_id))

            save_pickle(char_id_tuples_pickle_path, self.char_id_tuples)

    def _create_pos_id_tuples(self):
        pos_id_tuples_pickle_path = self._path_root + '_pos_idtuples.pickle'

        if os.path.exists(pos_id_tuples_pickle_path):

            print('Loading pos id'
                  ' tuples pickle {}'.format(pos_id_tuples_pickle_path))
            self.pos_id_tuples = load_pickle(pos_id_tuples_pickle_path)

        else:
            print('Creating pos id tuples pickle '
                  '{}'.format(pos_id_tuples_pickle_path))

            for tupl in self.tuples:
                premise = tupl[0]
                hypothesis = tupl[1]
                premise_pos_ids = self.lang.sentence2posindex(premise)
                hypothesis_pos_ids = self.lang.sentence2posindex(hypothesis)
                self.pos_id_tuples.append((premise_pos_ids,
                                           hypothesis_pos_ids))

            save_pickle(pos_id_tuples_pickle_path, self.pos_id_tuples)

    def get_batch_iterator(self, batch_size, use_char_embeddings=False,
                           shuffle=False, pos_tags=False, align_right=False,
                           cuda=False, context_window_size=1, allowed_labels=(),
                           double_hypotheses=False, differentiate_inputs=False,
                           max_prem_len=200):

        char_id_tuples = None
        pos_id_tuples = None
        allowed_label_ids = [constants.SNLI_LABEL_DICT[label]
                             for label in allowed_labels]
        if use_char_embeddings:
            char_id_tuples = self.char_id_tuples

        if pos_tags:
            if not self.lang.has_pos_tags:
                raise Exception("You provided the pos_tags flag but the "
                                "corpus does not have pos tags")
            pos_id_tuples = self.pos_id_tuples

        return BatchIterator(self.id_tuples,
                             batch_size,
                             char_id_tuples=char_id_tuples,
                             pos_id_tuples=pos_id_tuples,
                             shuffle=shuffle,
                             align_right=align_right,
                             cuda=cuda,
                             context_window_size=context_window_size,
                             allowed_label_ids=allowed_label_ids,
                             double_hypotheses=double_hypotheses,
                             differentiate_inputs=differentiate_inputs,
                             max_prem_len=max_prem_len)


SNLICorpus = type("SNLICorpus", (NLICorpus, ),
                  {'path': SNLI_CORPUS_PATH,
                  'label_dict': constants.SNLI_LABEL_DICT})

MultiNLICorpus = type("MultiNLICorpus", (NLICorpus, ),
                      {'path': MULTINLI_CORPUS_PATH,
                      'label_dict': constants.SNLI_LABEL_DICT})

FullNLICorpus = type("FullNLICorpus", (NLICorpus, ),
                     {'path': FULLNLI_CORPUS_PATH,
                     'label_dict': constants.SNLI_LABEL_DICT})
