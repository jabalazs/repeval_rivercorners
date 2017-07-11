# coding: utf-8
import os
import random
import shutil
import json
from repeval.corpus.snli_corpus import (SNLI_CORPUS_PATH,
                                        MULTINLI_CORPUS_PATH,
                                        FULLNLI_CORPUS_PATH,
                                        get_filename)

SNLI_PROPORTION_TO_SAMPLE = 0.15

if not os.path.exists(MULTINLI_CORPUS_PATH):
    print("MultiNLI corpus not found.")
    exit()

if not os.path.exists(SNLI_CORPUS_PATH):
    print("SNLI corpus not found. Download and extract it first: "
          "https://nlp.stanford.edu/projects/snli/snli_1.0.zip")
    exit()

os.mkdir(FULLNLI_CORPUS_PATH)

snli_train_filepath = get_filename(SNLI_CORPUS_PATH, "train")

multinli_train_filepath = get_filename(MULTINLI_CORPUS_PATH, "train")
multinli_dev_filepath = get_filename(MULTINLI_CORPUS_PATH, "dev")

fullnli_train_filepath = get_filename(FULLNLI_CORPUS_PATH, "train")
fullnli_dev_filepath = get_filename(FULLNLI_CORPUS_PATH, "dev")


with open(snli_train_filepath) as infile:
    snli_lines = infile.readlines()

# To make sure we don't sample any "-" labeled examples
filtered_snli_lines = []
for line in snli_lines:
    example = json.loads(line)
    if example['gold_label'] == '-':
        continue
    filtered_snli_lines.append(line)

sample_size = int(SNLI_PROPORTION_TO_SAMPLE * len(filtered_snli_lines))
snli_sample = random.sample(filtered_snli_lines, k=sample_size)

with open(multinli_train_filepath) as infile:
    multinli_lines = infile.readlines()

fullnli_lines = multinli_lines + snli_sample

with open(fullnli_train_filepath, 'w') as outfile:
    outfile.writelines(fullnli_lines)

shutil.copy(multinli_dev_filepath,
            fullnli_dev_filepath)
