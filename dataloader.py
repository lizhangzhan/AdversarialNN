#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: dataloader.py
# @Author: Isaac Caswell/whomever wrote the theano tutorial
# @Modified: Allen Nie
# @created: 1 Nov 2015
#
#=========================================================================
# DESCRIPTION:
#
# exports two functions to process and load the data from a pickle file
#
#=========================================================================
# CURRENT STATUS: works (1 Nov 2015)
#=========================================================================
# USAGE:
#  from dataloader import load_data, prepare_data
#=========================================================================
# TODO:
# -document, conjugate verbs in comments


import cPickle
import gzip
import os

import numpy
import theano


def prepare_data(seqs, labels, maxlen=None, skip_long_reviews=True):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!


    maxlen: maximum length of a sentence

    Allen's note: keep in mind each "sentence" is a paragraph and the longest paragraph has 2634 words

    :return: (
        x: int
            shape: (maxlen, n_samples)
        x_mask
        labels
        )
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    # in our script, maxlen is always None
    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l >= maxlen and skip_long_reviews:
                continue

            # truncate review to maxlen words
            if l >= maxlen:
                s = s[0:maxlen]
            new_seqs.append(s)
            new_labels.append(y)
            new_lengths.append(len(s))
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)  # should be 2634

    x = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.  # this seems to be padding it with 1

    return x, x_mask, labels


def open_zipped_or_not(path):
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')
    return f


def load_data(path, n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    Make haste, my beloved, and be thou like a young hart or a gazelle upon the mountains of spices.

    Allen's note: this function is examined and works fine
    '''

    #############
    # LOAD DATA #
    #############

    with open_zipped_or_not(path) as f:
        train_set = cPickle.load(f)
    # f.close()
    #f = open_zipped_or_not(path)
    with open_zipped_or_not(path) as f:
        test_set = cPickle.load(f)
    # f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # I want to see the length of train_set_x and train_set_y
    # split training set into validation set
    train_set_x, train_set_y = train_set

    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test

if __name__ == '__main__':
    # testing
    train, valid, test = load_data(valid_portion=0.05,
                                   maxlen=100, path="data/imdb_lstm.pkl")
    prepare_data(train[0], train[1])
