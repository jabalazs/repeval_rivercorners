
# Repeval 2017 Shared Task Code Submission

This repo contains the code developed for the [RepEval 2017 Shared Task](https://repeval2017.github.io/shared/) by the the team Rivercorners.

You can read our report "__Refining Raw Sentence Representations for Textual Entailment Recognition via Attention__" [here](https://www.arxiv.org).

## Installation instructions

1. Set the `DATA_PATH` global variable in `repeval/constants.py`. This directory will contain subdirectories with different kinds of data (corpora, embeddings, etc.)
2. Set the `DATABASE_PATH` global variable in `repeval/constants.py`.
   A sqlite3 database will be automatically created in that path
   for storing run information
3. Create the directory `corpus` in `DATA_PATH`
4. Download the data from http://www.nyu.edu/projects/bowman/multinli/multinli_0.9.zip
5. Unzip the `multnli_0.9.zip` file in `DATA_PATH/corpus/`
6. Access the directory `multnli_0.9/`
7. Remove the `_matched` or `_mismatched` suffix from the dev file
   you want to use as validation. The resulting dev file should be
   named `multinli_0.9_dev.jsonl`
8. Create the directory `word_embeddings` in `DATA_PATH/`
9. Download the 840B 300d Glove embeddings from
   http://nlp.stanford.edu/data/glove.840B.300d.zip and extract the
   contents in `DATA_PATH/word_embeddings/`
10. Make sure you have [pytorch](http://pytorch.org/) for python 2.7 installed in your environment (we used pytorch 0.1.12).
11. Also install the packages `tqdm` for displaying progress bars and `dataset` for interfacing with the sqlite database. You can run `pip install -r requirements.txt` to install them automatically
12. Run the following command in the topmost directory of the package:
   `python run_inner_att_model.py` for training with default parameters

You can also run `python run_inner_att_model.py --help` for displaying a list of available hyperparameters. There are some experimental features such as `--context_window_size` and `--pos_tags` that will not work well in conjunction with other settings, but these are not necessary to replicate our results.

The first time `run_inner_att.py` is run it will create some pickle files in `DATA_PATH/corpus/multinli_0.9/` for faster access in later runs. This will take some time.

Also, loading the Glove embeddings for the first time will take 10 minutes or so.

### Optional
If you want to combine Stanford's SNLI corpus with the MultiNLI corpus follow these instructions:

1. Download the SNLI corpus from https://nlp.stanford.edu/projects/snli/snli_1.0.zip
2. Unzip `snli_1.0.zip` in `DATA_PATH/corpus/`
3. Modify the `SNLI_PROPORTION_TO_SAMPLE` variable in `create_fullnli_dataset.py` to fit your needs. The default is 0.15 to match the other genres' proportion. We did this following what the authors of the MultiNLI corpus did ([link](https://arxiv.org/abs/1704.05426) to their paper):

    > We train models on SNLI, on MultiNLI, and on a mixture of both corpora. In the mixed setting, we use the full MultiNLI training set but down-sample SNLI by randomly selecting 15% of the SNLI training set at each epoch. This ensures that each available genre is seen with roughly equal frequency during training.

    Note that they sampled from the SNLI corpus at each epoch while we only do it once before beginning the training procedure.

4. Execute `python create_fullnli_dataset.py`. This will create the new directory `DATA_PATH/corpus/fullnli/` following structure similar to the original `DATA_PATH/corpus/multinli_0.9/`
5. Once these steps are completed you can choose which corpus to use when running your experiments, e.g. `python run_inner_att_model.py --corpus FullNLICorpus`

## Dependencies

This code is intended to be run with python 2.7. It uses the following libraries:

* pytorch >= 0.1.12
* tqdm >= 4.11.2
* dataset >= 0.8.0

## License
[MIT](LICENSE)

We also borrowed the Tree class from the [nltk](http://www.nltk.org/) package (`repeval/corpus/tree.py`) licensed under the [Apache License-2.0](https://www.apache.org/licenses/LICENSE-2.0).
