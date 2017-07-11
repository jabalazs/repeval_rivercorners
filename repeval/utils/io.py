try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import json
import csv
import dataset as dt
from repeval import constants
from datetime import datetime


def load_pickle(path):
    with open(path, 'rb') as f:
        pckl = pickle.load(f)
    return pckl


def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Pickle saved.')
    return None


def write_hyperparams(log_dir, params, mode='FILE'):
    if mode == 'FILE' or mode == 'BOTH':
        os.makedirs(log_dir)
        hyperparam_file = os.path.join(log_dir, 'hyperparams.json')
        with open(hyperparam_file, 'w') as f:
            f.write(json.dumps(params))
    if mode == 'DATABASE' or mode == 'BOTH':
        db = dt.connect(constants.DATABASE_CONNECTION_STRING)
        runs_table = db['runs']
        runs_table.insert(params)
    if mode not in ('FILE', 'DATABASE', 'BOTH'):
        raise ValueError('{} mode not recognized. Try with FILE, DATABASE or BOTH'.format(mode))


def update_in_db(datadict):
    db = dt.connect(constants.DATABASE_CONNECTION_STRING)
    runs_table = db['runs']
    runs_table.update(datadict, keys=['hash'])


def write_metrics(log_dir, params):
    """This code asumes the log_dir directory already exists"""
    metrics_file = os.path.join(log_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        f.write(json.dumps(params))


def write_output(log_dir, output, best_or_last, mode):
    dev_output_file_path = os.path.join(log_dir,
                                        best_or_last +
                                        '_' + mode + '.output.json')

    with open(dev_output_file_path, 'w') as f:
        f.write(json.dumps(output))


def write_probs(filepath, sentences_ids, probs):
    """probs: a list of torch.cuda.FloatTensor"""
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        for sent_id, prob_tensor in zip(sentences_ids, probs):
            line = [sent_id] + ['{:.12f}'.format(elem) for elem
                                in prob_tensor.tolist()]
            writer.writerow(line)


def _repr2str(sent_id, sent_repr, p_or_h):
    formatted_prem_repr = ['{:.12f}'.format(elem) for elem
                           in sent_repr.tolist()]
    prem_list_string = ' '.join(formatted_prem_repr)
    line = '\t'.join([sent_id, p_or_h])
    line = '\t'.join([line, prem_list_string])
    return line


def write_sent_reprs(filepath, sentences_ids, prem_reprs, hypo_reprs):
    """[]_reprs are lists of torch.cuda.FloatTensor"""
    with open(filepath, 'w') as f:
        for idx, sent_id in enumerate(sentences_ids):
            prem_repr = prem_reprs[idx]
            hypo_repr = hypo_reprs[idx]
            line_p = _repr2str(sent_id, prem_repr, 'p')
            line_h = _repr2str(sent_id, hypo_repr, 'h')
            f.write(line_p + '\n')
            f.write(line_h + '\n')


def get_hyperparams_from_model(model):
    """This assumes the hyperparams.json file is located in the same directory
    as the model"""
    dirname = os.path.dirname(model)
    with open(os.path.join(dirname, 'hyperparams.json'), 'r') as f:
        hyperparams = json.loads(f.read())
    return hyperparams


def get_hash_from_model(model):
    """This assumes the hyperparams.json file is located in the same directory
    as the model"""
    hyperparams = get_hyperparams_from_model(model)
    model_hash = hyperparams['hash']
    return model_hash


def get_datetime_from_model(model):
    """This assumes the hyperparams.json file is located in the same directory
    as the model"""
    hyperparams = get_hyperparams_from_model(model)
    run_datetime = hyperparams['datetime']
    run_datetime = run_datetime.replace('-', '_').replace(' ', '_').replace(':', '_')
    return run_datetime
