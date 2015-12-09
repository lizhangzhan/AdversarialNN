#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: generic_util.py
# @Author: Isaac Caswell
# @created: 21 February 2015
#
#===============================================================================
# DESCRIPTION:
#
# A file containing various useful functions I often find I have to write in my
# scripts/have to look up from other files.  For instance, how to plot things
# with legends, print things in color, and get command line arguments
#
#===============================================================================
# CURRENT STATUS: Works!  In progress.
#===============================================================================
# USAGE:
# import util
# util.colorprint("This text is flashing in some terminals!!", "flashing")
#
#===============================================================================
# CONTAINS:
#
#-------------------------------------------------------------------------------
# COSMETIC:
#-------------------------------------------------------------------------------
# colorprint: prints the given text in the given color
# time_string:
#       returns a string representing the date in the form '12-Jul-2013' etc.
#       Handy naming of files.
#-------------------------------------------------------------------------------
# FOR (LARGE) FILES:
#-------------------------------------------------------------------------------
# randomly_sample_file: given the name of some unnecessarily large file that you
#       have to work with, original_fname, randomly samples it to have a given
#       number of lines.  This function is used for when you want to do some
#       testing of your script on a pared down file first.
# scramble_file_lines:
#       randomly permutes the lines in the input file.  If the input
#       file is a list, permutes all lines in the iput files in the asme way.
#       Useful if you are doing SGD, for instance.
# file_generator:
#       streams a file line by line, and processes that line as a list of integers.
# split_file: given the name of some unnecessarily large file that you have to
#       work with, original_fname, this function splits it into a bunch of
#       smaller files that you can then do multithreaded operations on.
#
#===============================================================================
# TODO:
# make general plotting function


#standard modules
import numpy as np
import time
from collections import Counter, defaultdict
import heapq
# import matplotlib.pyplot as plt
import argparse
import shutil
import csv
import os
import re
import collections
import json
import gzip
import cPickle


import theano.tensor as T


#===============================================================================
# FUNCTIONS
#===============================================================================



def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)

    copied from theano dataset
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    # data_dir, data_file = os.path.split(dataset)
    # if data_dir == "" and not os.path.isfile(dataset):
    #     # Check if dataset is in the data directory.
    #     new_path = os.path.join(
    #         os.path.split(__file__)[0],
    #         "..",
    #         "data",
    #         dataset
    #     )
    #     if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
    #         dataset = new_path

    # if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
    #     import urllib
    #     origin = (
    #         'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    #     )
    #     print 'Downloading data from %s' % origin
    #     urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an np.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #np.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval



#-----------------------------------------------------------------------------------------

def colorprint(message, color="rand", newline=True):
    """
    '\033[%sm' + message +'\033[0m',
    """
    message = str(message)
    message = unicode(message)
    """prints your message in pretty colors! So far, only a few color are available."""
    if color == 'none': print message,
    if color == 'demo':
        for i in range(99):
            print '\n%i-'%i + '\033[%sm'%i + message + '\033[0m\t',
    print '\033[%sm'%{
        'neutral' : 99,
        'flashing' : 5,
        'underline' : 4,
        'red_highlight' : 41,
        'green_highlight' : 42,
        'orange_highlight' : 43,
        'blue_highlight' : 44,
        'magenta_highlight' : 45,
        'teal_highlight' : 46,
        'pink_highlight' : 46,
        'pink' : 35,
        'purple' : 34,
        'peach' : 37,
        'yellow' : 93,
        'blue' : 94,
        'teal' : 96,
        'rand' : np.random.choice(range(90, 98) + range(40, 48)+ range(30, 38)+ range(0, 9)),
        'green' : 92,
        'red' : 91,
        'bold' : 1
    }.get(color, 1)  + message + '\033[0m',
    if newline:
        print '\n'

def madness(delay=3):
    import threading
    import os
    def speaker():
        try:
            os.system("osascript -e \"set Volume 5\"")
            os.system("sleep %s"%delay)
            os.system("say -v whisper -r 1 Oh God &")
            os.system("sleep 3")
            os.system("osascript -e \"set Volume 3\"")
            os.system("say -v whisper Oh God &")
            os.system("sleep 2")
            os.system("say -v whisper help &")
            os.system("sleep 1")
            os.system("osascript -e \"set Volume 7\"")
            os.system("say -v veena will someone help him? &")
            os.system("sleep 4")

            os.system("osascript -e \"set Volume 5\"")
            os.system("say -v kyoko こんにちは、私の名前はKyokoです。日本語の音声をお届けします &")
            os.system("say -v tarik مرحبًا اسمي Tarik. أنا عربي من السعودية &")
            os.system("sleep 1")
            os.system("say -v lekha What kind of balls, does dees man have.")

            os.system("say -v whisper oh god Oh God Oh God &")
            os.system("say -v veena oh god.  oh god &")
            os.system("sleep 20")
            os.system("osascript -e \"set Volume 5\"")
            os.system("say -v whisper please &")
            os.system("sleep 1")
            os.system("say -v tarik مرحبًا اسمي &")
            # os.system("sleep 200")
            os.system("osascript -e \"set Volume 2\"")
            # os.system("say -v veena its friday friday friday ooh.  Gotta get down on friday &")
            # os.system("say -v tarik مرحبًا اسمي &")
            os.system("sleep 200")
            os.system("osascript -e \"set Volume 10\"")
            os.system("say -v milena Союз нерушимый республик свободных Сплотила навеки Великая Русь! Да здравствует созданный волей народов Единый, могучий Советский Союз! Славься, Отечество наше свободное, Дружбы народов надёжный оплот! Партия Ленина — сила народная Нас торжеству коммунизма ведёт! Сквозь грозы сияло нам солнце свободы, И Ленин великий нам путь озарил: На правое дело он поднял народы, На труд и на подвиги нас вдохновил! В победе бессмертных идей коммунизма Мы видим грядущее нашей страны, И Красному знамени славной Отчизны Мы будем всегда беззаветно верны! &")
            import webbrowser as wb
            os.system("sleep 5")
            wb.open("https://www.youtube.com/watch?v=x72w_69yS1A")


        except:
            pass

    th = threading.Thread(target=speaker)
    th.start()


def time_string(precision='day'):
    """ returns a string representing the date in the form '12-Jul-2013' etc.
    intended use: handy naming of files.
    """
    t = time.asctime()
    precision_bound = 10 #precision == 'day'
    yrbd = 19
    if precision == 'minute':
        precision_bound = 16
    elif precision == 'second':
        precision_bound = 19
    elif precision == 'year':
        precision_bound = 0
        yrbd = 20
    t = t[4:precision_bound] + t[yrbd:24]
    t = t.replace(' ', '-')
    return t


def random_string_signature(length = 4):
    candidates = list("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890")
    np.random.shuffle(candidates)
    return "".join(candidates[0:length])

#===============================================================================
# DEMO SCRIPT
#===============================================================================

if __name__ == '__main__':
    colorprint("nyaan", color="demo")
    madness(0)

