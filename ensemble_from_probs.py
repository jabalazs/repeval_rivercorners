#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import csv
import numpy as np

from repeval.constants import SNLI_LABEL_DICT

parser = argparse.ArgumentParser(
                   description='Obtain predicted classes by averaging '
                               'several softmax outputs.')

parser.add_argument('files', type=str, nargs='+',
                    help='prob files from which to generate the predictions')


def get_ids_and_ndarray_from_prob_file(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
        numeric_probs = [[float(elem) for elem in str_probs[1:]] for str_probs in lines]
        ids = [elem for str_probs in lines for elem in str_probs[:1]]
    return ids, np.array(numeric_probs)


def get_mean_from_array_list(array_list):
    """array list, a list of numpy arrays of dim
       (num_examples, num_classes)
       """
    tensor = np.stack(array_list, axis=2)
    mean = np.mean(tensor, axis=2)
    return mean


def get_label_ids_from_probs(probs):
    """probs is a numpy array of dimensions (num_examples, num_classes)
    return a list of length (num_examples)"""
    label_ids = np.argmax(probs, axis=1).tolist()
    return label_ids


def get_labels_from_label_ids(label_ids):
    label_id2name = {v: k for k, v in SNLI_LABEL_DICT.iteritems()}
    labels = [label_id2name[label_id] for label_id in label_ids]
    return labels


def run_main():
    args = parser.parse_args()
    print args
    filenames = args.files
    array_list = []
    for filename in filenames:
        ids, probs_array = get_ids_and_ndarray_from_prob_file(filename)
        array_list.append(probs_array)
    mean_probs = get_mean_from_array_list(array_list)
    label_ids = get_label_ids_from_probs(mean_probs)
    labels = get_labels_from_label_ids(label_ids)

    preds_filename = 'ensembled_predictions.csv'
    print('Writing {}'.format(preds_filename))
    with open(preds_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['pairID', 'gold_label'])
        for sent_id, pred_label in zip(ids, labels):
            writer.writerow([sent_id, pred_label])


if __name__ == "__main__":
    run_main()
