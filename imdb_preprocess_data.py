#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: process_data.py
# @Author: Allen Nie
# @created: 21 Nov 2015
#
# =========================================================================
# DESCRIPTION:
#
# based on process_data.py from CNN_sentence
# but largely adapted for IMDB corpus
# also combined with imdb_process_data.py from
# Theano LSTM
#
# This has two options: prepare corpus for LSTM
# and prepare corpus for CNN
# we don't set for validation (dataloader.py will do this)
#
# =========================================================================
# CURRENT STATUS:
# usable
# =========================================================================
# USAGE:
#  python imdb_process_data.py
# =========================================================================

from gensim.models import word2vec
import numpy as np
from collections import defaultdict
import sys, re
import theano
import cPickle as pkl
import pandas as pd
from nltk.tokenize import word_tokenize
import glob
import os


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def tokenize(text):
    """
    Modified tokenize() to use NLTK's tokenizer instead of moses
    This is for the IMDB corpus as well as CNN

    :param text: it takes in a string
    :type: string

    :rtype: [string]
    """
    tok_text = word_tokenize(clean_str(text))
    return tok_text


def tokenize_all(sentences):
    """

    :param sentences: same as tokenize, but this time a list of strings
    :type: [string]

    :rtype: [[string]]
    """
    tokenized = []
    print 'Tokenizing..'
    for sentence in sentences:
        tokenized.append(word_tokenize(clean_str(sentence)))
    print 'Done'
    return tokenized


def build_vocab(data_path, cv=10):
    """
    similar to build_dict() and part of build_data_cv()
    :param data_path: folder to IMDB high-level folder
    :return: corpus
    :rtype: [{"y", "text", "split"}]
    """
    corpus = []
    vocab = defaultdict(float)

    print 'Building vocab dictionary..'

    currdir = os.getcwd()
    os.chdir('%s/pos/' % data_path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = f.readline().strip()
            words = tokenize(sentence)
            for word in words:
                vocab[word] += 1
            data = {
                "y": 1,
                "text": sentence
            }
            corpus.append(data)
    os.chdir('%s/neg/' % data_path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = f.readline().strip()
            words = tokenize(sentence)
            for word in words:
                vocab[word] += 1
            data = {
                "y": 0,  # labels are inserted manually
                "text": sentence
            }
            corpus.append(data)
    os.chdir(currdir)

    return corpus, vocab


def load_test_data(data_path):
    """
    Similar to grab_data() but this is more for
    CNN program (fitting the format)

    We load it without generating a vocab
    and 0 represents <UNK> words
    :return: [{"y", "text"}]
    """
    corpus = []

    print 'Loading test data..'

    currdir = os.getcwd()
    os.chdir('%s/pos/' % data_path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = f.readline().strip()
            words = tokenize(sentence)
            data = {
                "y": 1,
                "text": sentence,
            }
            corpus.append(data)
    os.chdir('%s/neg/' % data_path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentence = f.readline().strip()
            words = tokenize(sentence)
            data = {
                "y": 0,  # labels are inserted manually
                "text": sentence,
            }
            corpus.append(data)
    os.chdir(currdir)

    return corpus


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    This method only loads vectors of known words into word_vecs
    """
    model = word2vec.Word2Vec.load_word2vec_format(fname, binary=True)
    word_vecs = {}

    # a loop over vocab, find it in word2vec model
    # only return found words
    # model.__get_item__ returns float32 type array
    for word in vocab:
        if word in model:
            word_vecs[word] = model[word]

    return word_vecs


def grab_data(path, word_idx_map):
    """
    If for test set, a word is unseen in word_idx_map
    we mark it as 0 (0 position in word_idx_map is reserved already)

    Hmmm, should I add unseen words of testing set to idx_map ??????

    :param dictionary: this is supposed to be word_idx_map we got
    :return:
    """
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = tokenize_all(sentences)

    seqs = [None] * len(sentences)  # so each seq represent a whole txt file
    for i, tokens in enumerate(sentences):
        seqs[i] = [word_idx_map[t] if t in word_idx_map else 0 for t in tokens]

    return seqs


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i

    :return: return a [[int]], index maps to vocab index and
             a modified vocab (index_list), with 0 mapping to (0,0,0..)
    """
    vocab_size = len(word_vecs)
    word_idx_map = defaultdict()  # this one remaps index, leaving 0 empty

    W = np.zeros(shape=(vocab_size + 1, k))  # word vector
    W[0] = np.zeros(k, dtype='float32')  # first one is all 0, mapping to unknown? or bias?
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]  # W[i], ith location is ith word's vector weight
        word_idx_map[word] = i  # kind of like vocab, map a word with an index
        i += 1
    return W, word_idx_map


def dump_LSTM_data(data_path, word_emb, word_idx_map):
    """
    This dumps a pkl file that fits LSTM's format
    Other than traditional train_x, train_y, test_x, test_y
    this also dumps word_emb in "imdb_lstm.pkl"
    and word_idx_map to "imdb_lstm.idxmap.pkl"
    :return:
    """
    # now we have word_idx_map (vocab dict) and W ready
    train_x_pos = grab_data(data_path + 'train/pos', word_idx_map)
    train_x_neg = grab_data(data_path + 'train/neg', word_idx_map)
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(data_path + 'test/pos', word_idx_map)
    test_x_neg = grab_data(data_path + 'test/neg', word_idx_map)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('imdb_lstm.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('imdb_lstm.idxmap.pkl', 'wb')
    pkl.dump((word_emb, word_idx_map), f, -1)  # dump both
    f.close()


def dump_CNN_data(corpus, word_emb, word_idx_map, vocab):
    """
    dump CNN-style data
    be careful because the test set needs to be dumped as well!!!
    Deprecated: DON'T USE THIS
    THIS IS NEVER USED. CNN loads LSTM's data dump

    :return:
    """
    f = open('imdb_cnn.pkl', 'wb')
    pkl.dump((corpus, word_emb, word_idx_map, vocab), f, -1)
    f.close()


if __name__ == '__main__':

    ARGS = sys.argv[1:]  # get rid of the first "filename.py" arg

    w2v_file = "/Users/Aimingnie/Documents/School/Stanford/CS 224N/DeepLearning/dataset/" \
               "GoogleNews-vectors-negative300.bin"
    data_path = "/Users/Aimingnie/Documents/School/Stanford/CS 224N/DeepLearning/dataset/aclImdb/"

    print "loading data..."

    # corpus is the same as revs in process_data.py
    # this only builds from the train set
    corpus, vocab = build_vocab(os.path.join(data_path, 'train'))
    max_l = np.max(pd.DataFrame(corpus)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(corpus))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",

    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    # unknown/unseen word (not seen in word2vec) is initialized uniformly
    add_unknown_words(w2v, vocab)

    # W is a list of words, with their embedding mapping (tailored to our vocab)
    # word_idx_map is a remapped vocab dictionary
    W, word_idx_map = get_W(w2v)

    if ARGS[0].lower() == 'lstm':
        dump_LSTM_data(data_path, W, word_idx_map)

    if ARGS[0].lower() == 'cnn':
        dump_CNN_data(corpus, W, word_idx_map, vocab)
