"""
Replica of https://github.com/yoonkim/CNN_sentence/blob/master/conv_net_sentence.py
"""

import cPickle as pkl
import os

import errno
import numpy as np
import sys
import warnings
import timeit
import re
from collections import defaultdict, OrderedDict
import numpy
import theano
import theano.tensor as T
from theano import pp
from theano.tensor.nnet import conv
from cnn_net_layers_classes import LeNetConvPoolLayer, MLPDropout
import argparse
import subprocess  # for printing to both std and log file

warnings.filterwarnings("ignore")

# Set the random number generators' seeds for consistency
SEED = 3435
numpy.random.seed(SEED)

LOG_FILE = None


def ReLU(x):
    y = T.maximum(0.0, x)
    return (y)


def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return (y)


def Tanh(x):
    y = T.tanh(x)
    return (y)


def Iden(x):
    y = x
    return (y)


# ============ Util function ==========

def init_log(prefix, LOG_FILE_LOC):
    """
    Initialize logging file (keep it open)
    :return:
    """
    global LOG_FILE
    if LOG_FILE_LOC is not None:
        if not os.path.exists(os.path.dirname(LOG_FILE_LOC)):
            try:
                os.makedirs(os.path.dirname(LOG_FILE_LOC))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # then we open/create that file
        LOG_FILE = open(prefix + "/" + LOG_FILE_LOC, 'wb')
    else:
        print "Writing log to stdout!"


def close_log():
    """
    Call this at the end of file to close log
    :return:
    """
    if LOG_FILE is not None:
        LOG_FILE.close()


def log(sent):
    print(sent)
    if LOG_FILE is not None:
        LOG_FILE.write(repr(sent) + " \n")


# ============= Parameters Getting ========

def get_args():
    """
    parses command line args"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', dest='PREFIX', type=str,
                        default='/Users/Aimingnie/Documents/School/Stanford/CS 224N/DeepLearning/dataset/',
                        help='path containing trailing /')
    parser.add_argument('--data', dest='DATAPATH', type=str,
                        default="imdb_lstm.pkl",
                        help="mostly it's a file named 'imdb_lstm.pkl', provide the abs path")
    parser.add_argument('--idxmap', dest='IDX_MAP_PATH', type=str,
                        default="imdb_lstm.idxmap.pkl",
                        help="this mostly is a file named 'imdb_lstm.idxmap.pkl', provide the abs path")

    parser.add_argument('--epochs', dest='N_EPOCHS', type=int, default=100)

    # ADVERSARIAL_EPSILON = .07
    # this CNN program doesn't have ADV_ALPHA. ADV_ALPHA is fixed by the learning rate
    # of the entire algorithm
    parser.add_argument('--adv_eps', dest='ADVERSARIAL_EPSILON', type=float, default=0.5)
    parser.add_argument('--adv_lr', dest='ADVERSARIAL_ALPHA', type=float, default=0.25)

    parser.add_argument('--adv', dest='ADVERSARIAL', action='store_true',
                        help="by default is set to 0 (false).  Running with --adv 1 flag sets it to true.")

    # learning rate for sgd, as well as adversarial
    parser.add_argument("--lr_decay", dest='LRDECAY', type=float, default=0.95)
    parser.add_argument("--maxlen", dest='MAXLEN', type=int, default=100,
                        help="the maximum length of words we'll take into account for each paragraph")
    parser.add_argument("--batchsize", dest='BATCHSIZE', type=int, default=16)

    # redirect writing streams
    parser.add_argument("--logfile", dest='LOG_FILE_LOC', type=str,
                        help="Specify the location of the output file")

    return parser.parse_args()


# ============= End ============

def get_layers(rng, x, y, Words, filter_hs, filter_shapes, pool_sizes,
               batch_size, img_h, img_w, conv_non_linear,
               hidden_units, feature_maps, activations, dropout_rate,
               non_static):
    """
    This represents the whole neural network
    structure that we built

    :param x: shared variable, of input
    :type x: theano.Shared T.matrix('x')

    :param Words: shared variable, word embedding
                  for adversarial examples, this one's values get updated

    :param filter_hs: same as passed to train_conv_net()
    :param filter_shapes: filter_shapes are appended inside train_conv_net

    :param pool_sizes: are also calculated inside train_conv_net()

    :param img_h: obtained inside train_conv_net()

    :param feature_maps: hidden_units[0], obtained inside train_conv_net()

    :return:
    """

    # similar to LeNet Tutorial
    # we boradcast(reshape) x matrix to a 4-D tensor
    # in LeNet
    # (batch_size, 1, MNIST_height, MNIST_width)
    # 1 = num of feature map (for original)
    # but what are we re-shaping??
    # Words.shape[1] = 300, word embedding size
    # x.shape[0] = total number of training sentences (images)
    # x.shape[1] = sentence vector length (0,0,0,24,25,654...)
    # (x.shape[1], Words.shape[1]) is the height and width of the sentence matrix
    layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))

    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]  # retrieving already stored filter_shapes
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                        image_shape=(batch_size, 1, img_h, img_w),
                                        filter_shape=filter_shape, poolsize=pool_size,
                                        non_linear=conv_non_linear)
        # we only keep 1st dim, collapse the rest
        # i.e., for (2,3,4,5) we get (2, 60)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)  # since we have 3 filters, we have 3 conv_layers (really??)
        layer1_inputs.append(layer1_input)

    # concatenate: for 2 matrices of (m x n), if concatenate, we get (m x n x 2)
    # we use tensor.concatenate((tensor.shape_padright(m1), tensor.shape_padright(m2)), axis = 2)
    # putting all the inputs together (here), all inputs have same dimension
    # WHAT!? IS THIS CONCATENATING A F*CKING LABEL TO THIS!?
    # I don't think it's concatenating a label.....but what is it??
    layer1_input = T.concatenate(layer1_inputs, 1)

    # we are overrding default hidden_units[0] value, which is 100
    # after filtering, we have 3 filters, we should have 300 feature maps
    hidden_units[0] = feature_maps * len(filter_hs)

    classifier = MLPDropout(rng, input=layer1_input,
                            layer_sizes=hidden_units, activations=activations,
                            dropout_rates=dropout_rate)

    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params

    # if non_static:
    #     # !!if word vectors are allowed to change, add them as model parameters
    #     # this is interesting...
    #     params += [Words]

    return classifier, conv_layers, params


def train_conv_net(datasets,
                   U,
                   img_w=300,
                   filter_hs=[3, 4, 5],
                   hidden_units=[100, 2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,  # not really using this param cause I'm lazy
                   n_epochs=150,  # maximum number of epochs
                   batch_size=16,
                   validFreq=370,
                   lr_decay=0.95,  # this is alpha
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,
                   adversarial=True,
                   adv_lr_rate=0.1,
                   adv_epsilon=0.25):
    """
    couple of modifications:
    datasets contain [train, valid, test], old one was [train, test]
    I don't know if U's format will work
    So, we train each iteration based on valid's performance on epoch
    kind of like LeNet from DeepLearning Tutorial

    Train a simple conv net
    U = word embedding
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    rng = np.random.RandomState(SEED)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # in old code, last elem of the array is y-label
    # in this code, y is always seperated
    # all the sentence lengths (in train, valid, test) are the same
    # old code has [0][0] - 1, because they are taking out the last elem: y-label
    img_h = len(train_set_x[0])  # length of the sentence
    filter_w = img_w  # vector

    # classic LeNet from Tutorial only has 20, 50 feature maps
    # we have more here
    # feature maps, i.e., 3 x 24 x 24  http://neuralnetworksanddeeplearning.com/chap6.html
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []

    # this is only for one layer
    for filter_h in filter_hs:
        # this is for filter_shape
        # (num_feature_maps at m, num_feature_maps at m-1, filter height, f width)
        # first/original input is by itself one feature map
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        # DeppLearning Tutorial just chose 2, here we adapt to sentence matrix
        # sentence matrix = (word_length, w2vec_vector_dim = 300)
        # -----------
        # This 0.23, 0.45 ...
        # is   0.56, 0.44 ...
        pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))

    parameters = [("adversarial", adversarial), ("adv_lr_rate", adv_lr_rate), ("adv_epsilon", adv_epsilon),
                  ("image shape", img_h, img_w), ("filter shape", filter_shapes), ("hidden_units", hidden_units),
                  ("dropout", dropout_rate), ("batch_size", batch_size), ("non_static", non_static),
                  ("learn_decay", lr_decay), ("conv_non_linear", conv_non_linear),
                  ("non_static", non_static), ("sqr_norm_lim", sqr_norm_lim), ("shuffle_batch", shuffle_batch)]

    log(parameters)

    # define model architecture
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # Those are word embedding
    # since we are modifying them (set_subtensor)
    # we can't flag borrow=True
    Words = theano.shared(value=U, name="Words")

    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)  # 300-dim 0 vector

    # update Word embedding into T.set_subtensor()
    # Words[0, :] since Words is a shared variable
    # [0, :] returns a sub-tensor: Subtensor{int64, ::}.0

    set_zero = theano.function([zero_vec_tensor],
                               updates=[(Words,
                                         T.set_subtensor(Words[0, :], zero_vec_tensor))],
                               allow_input_downcast=True)

    # this is very bulky right now
    # because you don't have enough knowledge on Theano......
    classifier, conv_layers, params = get_layers(rng, x, y, Words,
                                                 filter_hs, filter_shapes,
                                                 pool_sizes, batch_size,
                                                 img_h, img_w, conv_non_linear,
                                                 hidden_units, feature_maps,
                                                 activations, dropout_rate, non_static)

    cost = classifier.negative_log_likelihood(y)
    # dropout_cost is from the LAST drop_out layer
    dropout_cost = classifier.dropout_negative_log_likelihood(y)

    # we generate adversarial examples on normal cost
    # but we optimize on dropout_cost

    if adversarial:
        leaf_grads = theano.tensor.grad(cost, wrt=Words)
        # we need to wrt word embedding Words (by Jon Gauthier)
        anti_example = theano.tensor.sgn(leaf_grads)
        adv_example = Words + adv_epsilon * anti_example  # lol, I strongly don't think this is "savable" by any means

        # below two lines aren't quite working
        # the cost calculated by that is not a scalar, thus can't be updated

        adv_example_grad = theano.gradient.disconnected_grad(
            adv_example)  # I think this is the J(theta, x + eta * sgn(partial-derivative on x over J(theta, x, y)))

        # instead of Words, we drop in adv_example
        # so both cost, and dropout_cost should be updated

        cost = adv_lr_rate * cost + (1 - adv_lr_rate) * T.sum(adv_example_grad)  # formula on paper
        # cost is a scalar, but apparently, adv_example_grad IS NOT!!!!!
        # This is the PROBLEM

        grad_updates = sgd_updates_adadelta(params, cost, lr_decay, 1e-6, sqr_norm_lim)

    else:

        grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    # datasets[0] is the training sample
    # instead of original's manual numpy permutation
    # we use get_minibatches_idx from LSTM to do shuffling

    n_train_batches = int(np.round(train_set_x.shape[0] / batch_size))
    n_val_batches = int(np.round(valid_set_x.shape[0] / batch_size))
    n_test_batches = int(np.round(test_set_x.shape[0] / batch_size))

    log("%d train examples" % train_set_x.shape[0])
    log("%d valid examples" % valid_set_x.shape[0])
    log("%d test examples" % test_set_x.shape[0])

    # sys.exit(0)

    train_shared_set_x, train_shared_set_y = shared_dataset((train_set_x, train_set_y))
    valid_shared_set_x, valid_shared_set_y = shared_dataset((valid_set_x, valid_set_y))
    test_shared_set_x, test_shared_set_y = shared_dataset((test_set_x, test_set_y))

    val_model = theano.function([index], classifier.errors(y),
                                givens={
                                    x: valid_shared_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: valid_shared_set_y[index * batch_size: (index + 1) * batch_size]
                                }, allow_input_downcast=True)

    test_model = theano.function([index], classifier.errors(y),
                                 givens={
                                     x: test_shared_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: test_shared_set_y[index * batch_size: (index + 1) * batch_size]
                                 }, allow_input_downcast=True)

    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                      x: train_shared_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_shared_set_y[index * batch_size:(index + 1) * batch_size]
                                  }, allow_input_downcast=True)

    # this is from CNN_sentence
    # define a test_model_all() function
    # however, how is this different from LeNet's solution, and using test_model?
    # CNN_sentence appears to be using
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(), dtype="int32")] \
        .reshape((test_size, 1, img_h, Words.shape[1]))

    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))

    # interesting, in LeNet this does not happen
    test_layer1_input = T.concatenate(test_pred_layers, 1)  # wait..isn't this adding labels?? Or not...
    test_y_pred = classifier.predict(test_layer1_input)
    test_error = T.mean(T.neq(test_y_pred, y))
    test_model_all = theano.function([x, y], test_error, allow_input_downcast=True)

    # start training over mini-batches

    log('... training')

    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validFreq = min(n_train_batches, patience / 2)  # we are setting our own validFreq

    epoch = 0
    best_iter = 0
    best_val_perf = 0
    val_perf = []
    test_perf = []
    cost_epoch = 0

    # early stop
    done_looping = False
    best_validation_loss = np.inf

    num_minibatch = 0  # this marks how many minibatches we have consumed

    start_time = timeit.default_timer()

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        # print("epoch: " + str(epoch))
        # we are always shuffling each batch
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):

                num_minibatch += 1

                # We can't calculate iteration like old way
                # because our minibatches are RANDOMly shuffled
                # so we use num_minibatch to replace index, which goes from 1 to end
                iter = (epoch - 1) * n_train_batches + num_minibatch

                # print("iter: " + str(iter))
                cost_ij = train_model(minibatch_index)  # this returns normal (no-dropout) cost..
                # that we don't even use
                set_zero(zero_vec)  # hmmmmm, what's this doing??

                if iter % 100 == 0:
                    log('training @ iter = %d' % iter)

                if (iter + 1) % validFreq == 0:
                    validation_losses = [val_model(i) for i
                                         in range(n_val_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    log('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    val_perf.append(this_validation_loss)

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss

                        best_val_perf = 1 - np.mean(best_validation_loss)
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in range(n_test_batches)
                            ]
                        test_score = np.mean(test_losses)
                        log(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                        test_perf.append(test_score)

                if patience <= iter:
                    done_looping = True
                    break

            num_minibatch = 0  # reset this to 0

    end_time = timeit.default_timer()

    test_losses = [
        test_model(i)
        for i in range(n_test_batches)
        ]
    final_test_score = np.mean(test_losses)

    log('Optimization complete.')
    log('Best validation score of %f %% obtained at epoch %i, '
          'with test performance %f %%' %
          (best_val_perf * 100., best_iter, final_test_score * 100.))
    log('The code for file ' +
          os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    -----------------------------------------------------
    :param int n: the number of examples in question
    returns a list fo tuples of the form
        (minibatch_idx, [list of indexes of examples])
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def shared_dataset(data_xy):
    """
    Store data in shared variable, so when running on GPU
    Theano will not copy it over to GPU for every use
    (This is taken from my lr_network.py)
    """
    data_x, data_y = data_xy
    # transform a normal Python array into numpy array with dtype
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(np.asarray(data_y, dtype='int32'), borrow=True)  # 'int32'
    return shared_x, shared_y


def sgd_updates_adadelta(params, cost,
                         rho=0.95, epsilon=1e-6,
                         norm_lim=9, word_vec_name="Words"):
    """
    adadelta (and AdaGrad) is not affected too much by learning rate.
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    # OrderedDict is just a list of tuples..eh.disgusting
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def padding_single(seq, max_l, filter_h=5):
    """
    We pad to the longest sentence
    so returned vector all has the same length
    (but training set and testing set have different length)
    :return:
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    x.extend(seq)

    while len(x) < max_l + 2 * pad:
        x.append(0)

    return x


def padding(seqs, max_l, filter_h=5):
    """
    :param seqs: list of vectors
    :param max_l: maximum length of a sentence/paragraph
    :param k: word vector
    :param filter_h: the highest width of a filter
    :return:
    """
    result = []
    for seq in seqs:
        result.append(padding_single(seq, max_l, filter_h))
    return result


def sentence_filtering(train_set, maxlen):
    """
    run this function before padding
    we filter out sentences that are longer than maxLen
    modeled similar to dataloader.py in LSTM
    :return:
    """
    new_train_set_x = []
    new_train_set_y = []
    for x, y in zip(train_set[0], train_set[1]):
        if len(x) < maxlen:
            new_train_set_x.append(x)
            new_train_set_y.append(y)
    train_set = (new_train_set_x, new_train_set_y)
    del new_train_set_x, new_train_set_y

    return train_set


def sentence_clipping(sentence_set, maxLen=200):
    """
    This clips the sentence to a max length
    originally length 2000 something is too slow to train on
    clip starts from the tail
    :param sentence_set: this should be something like train_set_x
    :param maxLen: 200
    :return:
    """
    new_set = []
    for sentence in sentence_set:
        new_set.append(sentence[:maxLen])
    return new_set


def load_data(path, valid_portion=0.1, filter_h=5, maxlen=200):
    """
    adapted from LSTM, read in like LSTM
    the only difference: we are doing wide-convolution, so need
    to add padding.

    :return: a list of vectors that is padded and has index as their [i]
    """

    #############
    # LOAD DATA #
    #############

    f = open(path)
    train_set = pkl.load(f)
    test_set = pkl.load(f)
    f.close()

    # we don't have maxlen here
    # split training set into validation set
    train_set_x, train_set_y = sentence_filtering(train_set, maxlen=maxlen)

    # get max length of training set
    lengths = [len(s) for s in train_set_x]
    train_maxlen = np.max(lengths)  # should be 2634 # but after filtering, it should be 100

    n_samples = len(train_set_x)  # after maxlen=100, we get 2444 samples
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    test_set_x, test_set_y = sentence_filtering(test_set, maxlen=maxlen)
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    test_lengths = [len(s) for s in test_set_x]
    test_maxlen = np.max(test_lengths)

    # let's do padding (for filter) here
    train_set_x = padding(train_set_x, train_maxlen, filter_h)
    valid_set_x = padding(valid_set_x, train_maxlen, filter_h)
    test_set_x = padding(test_set_x, test_maxlen, filter_h)

    train = (np.asarray(train_set_x, dtype=theano.config.floatX),
             np.asarray(train_set_y, dtype='int32'))
    valid = (np.asarray(valid_set_x, dtype=theano.config.floatX),
             np.asarray(valid_set_y, dtype='int32'))
    test = (np.asarray(test_set_x, dtype=theano.config.floatX),
            np.asarray(test_set_y, dtype='int32'))

    return train, valid, test


def load_idx_map(path):
    f = open(path, 'rb')
    word_emb, word_idx_map = pkl.load(f)
    f.close()
    return word_emb, word_idx_map


if __name__ == '__main__':
    ARGS = get_args()

    pwd = os.path.dirname(os.path.realpath(__file__))

    print("initializing logs...")
    init_log(pwd, ARGS.LOG_FILE_LOC)

    log("loading data...")
    DATAPATH = ARGS.PREFIX + ARGS.DATAPATH

    IDX_MAP_PATH = ARGS.IDX_MAP_PATH

    train, valid, test = load_data(DATAPATH,
                                   valid_portion=0.1,
                                   filter_h=5,
                                   maxlen=ARGS.MAXLEN)
    word_emb, word_idx_map = load_idx_map(ARGS.PREFIX + IDX_MAP_PATH)

    log("data loaded!")

    # mode = sys.argv[1]
    non_static = False

    # if mode == "-nonstatic":
    #     print "model architecture: CNN-non-static"
    #     non_static = True
    # elif mode == "-static":
    #     print "model architecture: CNN-static"
    #     non_static = False
    execfile(pwd + "/cnn_net_layers_classes.py")

    # instead of cross-validation of 10-fold
    # We do a straight forward validation
    train_conv_net([train, valid, test],
                   word_emb,  # same as W
                   lr_decay=ARGS.LRDECAY,
                   # this is the "rho" value for adadelta, lr for adadelta is always 1.0, default = 0.95
                   filter_hs=[3, 4, 5],
                   conv_non_linear="relu",
                   hidden_units=[100, 2],
                   shuffle_batch=True,
                   n_epochs=ARGS.N_EPOCHS,
                   sqr_norm_lim=9,  # this is an adadelta parameter
                   non_static=non_static,
                   batch_size=ARGS.BATCHSIZE,
                   dropout_rate=[0.5],  # if we are using adv, we aren't optimizing based on drop_out cost
                   adversarial=ARGS.ADVERSARIAL,
                   adv_lr_rate=ARGS.ADVERSARIAL_ALPHA,  # learning rate for adversarial
                   adv_epsilon=ARGS.ADVERSARIAL_EPSILON)

    # proposal: what if we do this kind of random search
    # that we start a trial, original CNN runs on one set of params,
    # adv programs try a whole range of params and see if it can beat original CNN
    # obviously one thing it won't control is that original has drop_out...
    # but we NEED to beat drop_out....

    close_log()
