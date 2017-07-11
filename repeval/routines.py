# coding: utf-8
from __future__ import division
from __future__ import print_function

import socket
import hashlib
import json
import os
import csv
from datetime import datetime

from repeval.optim import OptimWithDecay
from repeval.utils.io import (write_hyperparams,
                              write_metrics,
                              write_probs,
                              write_sent_reprs,
                              write_output,
                              update_in_db,
                              get_hash_from_model,
                              get_datetime_from_model)

from repeval.constants import SNLI_LABEL_DICT

from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F


_RUN_DAY = datetime.now().strftime('%Y_%m_%d')
_RUN_TIME = datetime.now().strftime('%H_%M_%S')
EXECUTION_TIME = datetime.now()


def complete_params(params, Model):
    """Add some more iformation to the args not provided through argparse"""
    params['datetime'] = EXECUTION_TIME.strftime('%Y-%m-%d %H:%M:%S')
    params['host_name'] = socket.gethostname()
    params['model'] = Model.__name__
    try:
        with open('/etc/machine-id', 'r') as f:
            # get machine-id and remove newline feed from the end
            params['machine_id'] = f.read()[:-1]
    except IOError as e:
        print('Machine_id file not found. Creating run hash without it.'
              ' {}'.format(str(e)))
    return params


def _build_hash(params):
    """Build hash from params"""
    return hashlib.sha1(json.dumps(params, sort_keys=True)).hexdigest()


def write_output_details(output_save_path, corpus, pred_label_ids,
                         best_or_last, mode):
    idx2label = {idx: label for label, idx in corpus.label_dict.items()}
    idx2word = corpus.lang.index2word
    result = []
    for i, (id_tuple, pred_label_id) in enumerate(zip(corpus.id_tuples,
                                                      pred_label_ids)):
        prem_ids = id_tuple[0]
        hypo_ids = id_tuple[1]
        gold_label_id = id_tuple[2]
        pair_id = id_tuple[3]
        premise_words = map(idx2word.__getitem__, prem_ids)
        hypothesis_words = map(idx2word.__getitem__, hypo_ids)
        gold_label = idx2label[gold_label_id]
        pred_label = idx2label[pred_label_id]
        # Assumes the corpus is NOT shuffled
        genre = corpus.raw_examples[i].genre
        result.append((pair_id, premise_words, hypothesis_words, gold_label,
                       pred_label, genre))

    output = {"fields": ['pair_id', 'premise', 'hypothesis',
                         'reference_label', 'predicted_label', 'genre'],
              "result": result}

    write_output(output_save_path, output, best_or_last, mode)


def main(args, Model, Corpora, EmbeddingsList, loss_function):

    torch.manual_seed(args.seed)
    hyperparams = sorted(vars(args).iteritems())
    for k, v in hyperparams:
        print('{}: {}'.format(k, v))

    # https://discuss.pytorch.org/t/high-gpu-memory-demand-for-pytorch/669
    torch.backends.cudnn.benchmark = True

    RUN_SAVE_PATH = os.path.join(args.runs_dir, _RUN_DAY, _RUN_TIME)

    Corpus = Corpora[args.corpus]

    # building/loading model
    if args.load:
        print('Loading model from {}'.format(args.load))
        model = torch.load(args.load)
        train_corpus = None
    else:
        train_corpus = Corpus(mode="train", chars=args.use_char_embeddings)
        Embeddings = EmbeddingsList[args.embeddings]
        embeddings = train_corpus.lang.get_torch_embeddings(Embeddings)
        num_classes = train_corpus.num_classes
        char_embeddings = None
        pos_embeddings = None

        if args.use_char_embeddings:
            char_embeddings = train_corpus.lang.get_torch_char_embeddings(
                                           args.char_embedding_dim)

        if args.pos_tags:
            pos_embeddings = train_corpus.lang.get_torch_pos_embeddings(
                                                        args.pos_embedding_dim)

        print('Creating model with fresh parameters')

        model = Model(embeddings, num_classes, args, char_embeddings,
                      pos_embeddings)

        if args.param_init:
            if model.__class__.__name__ == "SelfAttentiveModel":
                for p in model.self_attentive_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)

            elif model.__class__.__name__ == "MultiInnerAtt":

                for p in model.prem_attention_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)

                for p in model.hypo_attention_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)

            elif model.__class__.__name__ in ('InnerAtt'):
                for p in model.attention_layer_p.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
                for p in model.attention_layer_h.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            else:
                for p in model.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)

        print(model)

    if args.cuda:
        model.cuda()

    if args.mode == 'train':
        if not train_corpus:
            train_corpus = Corpus(mode="train", chars=args.use_char_embeddings)

        align_right = True if args.invert_premises else False
        train_batches = train_corpus.get_batch_iterator(
                              args.batch_size,
                              cuda=args.cuda,
                              shuffle=True,
                              align_right=align_right,
                              use_char_embeddings=args.use_char_embeddings,
                              context_window_size=args.context_window_size,
                              pos_tags=args.pos_tags,
                              double_hypotheses=args.double_hypotheses,
                              differentiate_inputs=args.differentiate_inputs,
                              max_prem_len=200)

        dev_corpus = Corpus(mode="dev", chars=args.use_char_embeddings,
                            from_raw=True)
        dev_batches = dev_corpus.get_batch_iterator(
                                  args.batch_size,
                                  cuda=args.cuda,
                                  shuffle=False,
                                  use_char_embeddings=args.use_char_embeddings,
                                  context_window_size=args.context_window_size,
                                  pos_tags=args.pos_tags)

        args_dict = vars(args)
        args_dict = complete_params(args_dict, Model)
        RUN_HASH = _build_hash(args_dict)
        args_dict['hash'] = RUN_HASH
        write_hyperparams(RUN_SAVE_PATH, args_dict,
                          mode='BOTH' if args.update_db else 'FILE')
        num_batches = len(train_batches)
        print('Amount of batches: {}'.format(num_batches))

        progress_bar = tqdm(total=args.epochs * num_batches)
        progress_bar.set_description(model.__class__.__name__)

        optimizer = OptimWithDecay(model.parameters(),
                                   method=args.optim,
                                   initial_lr=args.lr,
                                   max_grad_norm=args.clip,
                                   lr_decay=args.lr_decay,
                                   start_decay_at=None)
        best_accuracy = None
        best_epoch = 0
        for epoch in range(args.epochs):

            train_epoch(train_batches, model, optimizer, loss_function, epoch,
                        progress_bar=progress_bar,
                        log_interval=args.log_interval)

            eval_mode = "dev"
            logits, mean_valid_loss, validation_accuracy, pred_label_ids = \
                evaluate(dev_batches, model, dev_corpus, loss_function, args,
                         eval_mode, progress_bar=progress_bar)

            if args.update_learning_rate:
                optimizer.update_learning_rate(mean_valid_loss, epoch)

            if not best_accuracy or best_accuracy < validation_accuracy:
                save_path = os.path.join(RUN_SAVE_PATH, 'best_model.pth')
                progress_bar.write('Saving model in {}'.format(save_path))
                torch.save(model, save_path)
                if args.write_output:
                    write_output_details(RUN_SAVE_PATH, dev_corpus,
                                         pred_label_ids,
                                         'best',
                                         eval_mode)
                best_accuracy = validation_accuracy
                best_epoch = epoch
                write_metrics(RUN_SAVE_PATH,
                              dict(best_valid_accuracy=best_accuracy))
                if args.update_db:
                    update_data = dict(hash=RUN_HASH, best_valid_acc=best_accuracy)
                    update_in_db(update_data)

            torch.save(model, os.path.join(RUN_SAVE_PATH, 'latest_model.pth'))
            if args.early_stopping_epochs:
                if epoch - best_epoch == args.early_stopping_epochs:
                    progress_bar.write('Validation accuracy did not improve '
                                       'in {} epochs, stopping early.'
                                       ''.format(args.early_stopping_epochs))
                    break

    # we run testing ALWAYS
    test_corpus = Corpus(mode="test", chars=args.use_char_embeddings,
                         from_raw=True)
    test_batches = test_corpus.get_batch_iterator(
                                  args.batch_size,
                                  cuda=args.cuda,
                                  shuffle=False,
                                  use_char_embeddings=args.use_char_embeddings,
                                  context_window_size=args.context_window_size,
                                  pos_tags=args.pos_tags,
                                  max_prem_len=None)

    (probs,
     mean_test_loss,
     test_accuracy,
     pred_label_ids_test) = evaluate(test_batches, model, test_corpus,
                                     loss_function, args,
                                     "test")

    if args.load:
        run_hash = get_hash_from_model(args.load)
        update_dict = dict(hash=run_hash, test_acc=test_accuracy)
        if args.update_db:
            update_in_db(update_dict)


def train_epoch(batches, model, optimizer, loss_function, epoch,
                progress_bar=None, log_interval=200):
    batch_loss = 0
    num_batches = len(batches)
    model.train()
    for batch_id in range(num_batches):
        model.zero_grad()
        pair_ids = batches[batch_id][0]
        premises_tuple = batches[batch_id][1]
        hypotheses_tuple = batches[batch_id][2]
        labels = batches[batch_id][3]
        char_premises_tuple = batches[batch_id][4]
        char_hypotheses_tuple = batches[batch_id][5]
        pos_premises_tuple = batches[batch_id][6]
        pos_hypotheses_tuple = batches[batch_id][7]

        if model.__class__.__name__ in ("SelfAttentiveModel",
                                        "MultiInnerAtt"):
            logits, prem_A, hypo_A = \
                model(premises_tuple, hypotheses_tuple,
                      char_premises_tuple, char_hypotheses_tuple,
                      pos_premises_tuple, pos_hypotheses_tuple)

            loss = loss_function(logits, labels, prem_A, hypo_A)

        elif model.__class__.__name__ in ("InnerAtt"):
            (logits,
             prem_repr,
             hypo_repr) = model(premises_tuple, hypotheses_tuple,
                                char_premises_tuple, char_hypotheses_tuple,
                                pos_premises_tuple, pos_hypotheses_tuple)
            loss = loss_function(logits, labels)

        else:
            logits = model(premises_tuple, hypotheses_tuple,
                           char_premises_tuple, char_hypotheses_tuple,
                           pos_premises_tuple, pos_hypotheses_tuple)

            loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        batch_loss += loss.data[0]

        # Logging
        if progress_bar:
            pbar = progress_bar
            if batch_id > 0 and batch_id % log_interval == 0:
                mean_loss = batch_loss / log_interval

                pbar.write('Epoch {}, batch {}, lr: {}, '
                           'mean loss: {}'.format(epoch, batch_id,
                                                  optimizer.lr, mean_loss))
                batch_loss = 0
                pbar.update(log_interval)


def evaluate(batches, model, corpus, loss_function, args, mode,
             progress_bar=None):
    model.eval()
    total_examples = len(corpus.id_tuples)
    total_batches = len(batches)

    correct_labels = 0
    num_classes = corpus.num_classes
    class_totals = torch.zeros(num_classes)
    class_corrects = torch.zeros(num_classes)
    class_refs = torch.zeros(num_classes)
    total_loss = 0
    pred_probs = []
    prem_reprs = []
    hypo_reprs = []
    pred_label_ids = []
    ref_label_ids = []
    sentences_ids = []

    for batch_id in range(total_batches):
        sent_ids = batches[batch_id][0]
        premises_tuple = batches[batch_id][1]
        hypotheses_tuple = batches[batch_id][2]
        labels = batches[batch_id][3]
        char_premises_tuple = batches[batch_id][4]
        char_hypotheses_tuple = batches[batch_id][5]
        pos_premises_tuple = batches[batch_id][6]
        pos_hypotheses_tuple = batches[batch_id][7]

        if model.__class__.__name__ in ("SelfAttentiveModel",
                                        "MultiInnerAtt"):
            logits, prem_A, hypo_A = model(premises_tuple, hypotheses_tuple,
                                           char_premises_tuple,
                                           char_hypotheses_tuple,
                                           pos_premises_tuple,
                                           pos_hypotheses_tuple)

            loss = loss_function(logits, labels, prem_A, hypo_A)

        elif model.__class__.__name__ in ("InnerAtt"):
            (logits,
             prem_repr,
             hypo_repr) = model(premises_tuple, hypotheses_tuple,
                                char_premises_tuple, char_hypotheses_tuple,
                                pos_premises_tuple, pos_hypotheses_tuple)

            loss = loss_function(logits, labels)
        else:
            logits = model(premises_tuple, hypotheses_tuple,
                           char_premises_tuple, char_hypotheses_tuple,
                           pos_premises_tuple, pos_hypotheses_tuple)

            loss = loss_function(logits, labels)

        total_loss += loss.data[0]

        # Obtain prediction label
        prob_distr = F.softmax(logits)
        _, pred_labels = torch.max(logits.data, 1)
        pred_labels = Variable(pred_labels, volatile=True).cuda().squeeze()
        correct_labels += (pred_labels.data == labels.data).sum()

        for idx, label in enumerate(labels):
            class_refs[labels.data[0]] += 1
            if (pred_labels.data == labels.data)[idx] == 1:
                class_corrects[pred_labels.data[idx]] += 1
            class_totals[pred_labels.data[idx]] += 1

        for i in range(len(pred_labels)):
            pred_label_ids.append(int(pred_labels.data[i]))
            ref_label_ids.append(int(labels.data[i]))
            sentences_ids.append(sent_ids[i])
            pred_probs.append(prob_distr.data[i])
            if model.__class__.__name__ in ("InnerAtt"):
                prem_reprs.append(prem_repr.data[i])
                hypo_reprs.append(hypo_repr.data[i])

    mean_valid_loss = total_loss / total_batches
    validation_accuracy = correct_labels / total_examples

    if progress_bar:
        print_f = progress_bar.write
    else:
        print_f = print
    print_f('\tMean validation loss: {}'.format(mean_valid_loss))
    print_f('\tCorrect labels: {}'.format(correct_labels))
    print_f('\tAccuracy: {}'.format(validation_accuracy))
    print_f('\tReal labels per class: '
            '{}'.format(list(class_refs.numpy().astype(int))))
    print_f('\tPredicted labels per class: '
            '{}'.format(list(class_totals.numpy().astype(int))))
    print_f('\tCorrect labels per class: '
            '{}'.format(list(class_corrects.numpy().astype(int))))

    # Write predictions and probabilities only if a model is loaded
    if (mode == 'test' and args.load
       and corpus.__class__.__name__ != 'QuoraCorpus'):
        # Write predictions
        model_run_datetime = get_datetime_from_model(args.load)
        preds_filename = './' + corpus.__class__.__name__ + '_preds_'
        preds_filename += model_run_datetime + '.csv'
        print('Writing predictions file in '
              '{}'.format(os.path.abspath(preds_filename)))
        label_id2name = {v: k for k, v in SNLI_LABEL_DICT.iteritems()}
        with open(preds_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['pairID', 'gold_label'])
            for sent_id, pred_label in zip(sentences_ids, pred_label_ids):
                writer.writerow([sent_id, label_id2name[pred_label]])

        # Write class probabilities
        probs_filename = './' + corpus.__class__.__name__ + '_probs_'
        probs_filename += model_run_datetime + '.csv'
        print('Writing class probabilities file in '
              '{}'.format(os.path.abspath(probs_filename)))
        write_probs(probs_filename, sentences_ids, pred_probs)

        # Write sentence representations
        sents_filename = './' + corpus.__class__.__name__ + '_sents_'
        sents_filename += model_run_datetime + '.csv'
        print('Writing sentence representations file in '
              '{}'.format(os.path.abspath(sents_filename)))
        write_sent_reprs(sents_filename, sentences_ids, prem_reprs, hypo_reprs)

        print('Writing output...')
        write_output_details('./', corpus,
                             pred_label_ids,
                             corpus.__class__.__name__ + '_best',
                             'test')

    if (mode == 'test' and args.load
       and corpus.__class__.__name__ == 'QuoraCorpus'):
        with open('predictions.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['test_id', 'is_duplicate'])
            for sent_id, pred_label in zip(sentences_ids, pred_label_ids):
                writer.writerow([sent_id, pred_label])
            writer.writerow(['379205', 1])
            writer.writerow(['817520', 1])
            writer.writerow(['943911', 1])
            writer.writerow(['1046690', 1])
            writer.writerow(['1270024', 1])
            writer.writerow(['1461432', 1])

    return pred_probs, mean_valid_loss, validation_accuracy, pred_label_ids
