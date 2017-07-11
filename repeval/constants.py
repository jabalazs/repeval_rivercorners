import os

DATA_PATH = ''
DATABASE_PATH = ''

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
NUM_ID = 4

PAD_TOKEN = '_PAD'
UNK_TOKEN = '_UNK'
SOS_TOKEN = '_SOS'
EOS_TOKEN = '_EOS'
NUM_TOKEN = '_NUM'

UNK_CHAR_ID = 0
UNK_CHAR_TOKEN = ' '

SNLI_LABEL_DICT = {'neutral': 0, 'contradiction': 1, 'entailment': 2}

# DATABASE PARAMETERS
_DB_NAME = 'repeval_runs.db'

DATABASE_CONNECTION_STRING = 'sqlite:///' + os.path.join(DATABASE_PATH,
                                                         _DB_NAME)
