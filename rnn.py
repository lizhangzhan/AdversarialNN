#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: rnn.py
# @Author: Isaac Caswell/whomever wrote the theano tutorial
# @created: 1 Nov 2015
#
#===============================================================================
# DESCRIPTION:
#
# trains and tests an rnn.  Can be given ots of options, like whether to use LSTM
#
#===============================================================================
# CURRENT STATUS: works (1 Nov 2015)
#===============================================================================
# USAGE:
#  python rnn.py --data toy_corpus --hdim 300 --epochs 2
#===============================================================================
# TODO:
# -document, conjugate verbs in comments
# document more!!
# maybe make into a class/decompose
# make models save to a folder called 'models'
#


from collections import OrderedDict
import cPickle as pkl
import sys
import time
import util
import argparse

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from dataloader import load_data, prepare_data
from rnn_util import *

#================================================================================

OPTIMIZERS = {
            "sgd": sgd,
             "adadelta": adadelta,
             "rmsprop": rmsprop
             }

#================================================================================
def get_args():
    """
    parses command line args"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', dest = 'DEBUG', default=False, action='store_true')
    parser.add_argument('--id', dest = 'ID', default='', type=str)
    parser.add_argument('--redirect', dest = 'REDIRECT_OUTPUT_TO_FILE', default=False, action='store_true')
    parser.add_argument('--wemb_init', dest = 'WEMB_INIT', type=str, default="word2vec", help="What word embeddings to initialize with.  May be one of {word2vec, random}.   Note that if you use this flag, your --wdim flag must agree with the dimensionality of these pretrained vectors.")


    parser.add_argument('--adv', dest = 'ADVERSARIAL', type=int, default= 0, help="by default, self.ADVERSARIAL is set to 0 (false).  Running with --adv 1 flag sets it to true.")#ADVERSARIAL = False
    parser.add_argument('--data', dest = 'DATANAME', type=str, default='imdb', help = "'imdb' or 'toy_corpus'")## DATASET = "./data/toy_corpus.pkl"

    parser.add_argument('--epochs', dest = 'MAX_EPOCHS', type=int, default=1000)#MAX_EPOCHS = 2
    parser.add_argument('--alpha', dest = 'ADVERSARIAL_ALPHA', type=float, default=0.25)#ADVERSARIAL_ALPHA = 0.9
    parser.add_argument('--eps', dest = 'ADVERSARIAL_EPSILON', type=float, default=0.5)#ADVERSARIAL_EPSILON = .07
    parser.add_argument('--encoder', dest = 'ENCODER', type=str, default='lstm', help="lstm or rnn_vanilla")#ENCODER = 'lstm' if 0 else 'rnn_vanilla'

    parser.add_argument("--hdim", dest='HIDDEN_DIM', type=int, default=125)  # dimensionality of the hidden layer
    parser.add_argument("--wdim", dest='WORD_DIM', type=int, default=300)  # dimensionality of the word embeddings
    parser.add_argument("--reg", dest='l2_reg_U', type=float, default=0.)  # Weight decay for the classifier applied to the U weights.
    parser.add_argument("--lrate", dest='LRATE', type=float, default=0.0001)  # Learning rate for sgd (not used for adadelta and rmsprop)
    parser.add_argument("--optimizer", dest='OPTIMIZER', default="adadelta")  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    parser.add_argument("--maxlen", dest='MAXLEN', type=int, default=100)  # Sequence longer then this get ignored
    parser.add_argument("--batch_size", dest = "BATCH_SIZE", type=int, default=16)  # The batch size during training.
    parser.add_argument("--weight-init", dest = "WEIGHT_INIT", type=str, default="ortho_1.0")  # The batch size during training.
    parser.add_argument("--clip", dest='GRAD_CLIP_THRESH', type=float, default=1.0)  # threshold for gradient clips
    parser.add_argument("--noise_std", dest = "NOISE_STD", type=float, default=0., help="damned if I know what this is")

    return parser.parse_args()
#================================================================================

class Rnn():
    def __init__(self,
            adversarial,
            adv_alpha = None,
            adv_epsilon = None,
            hidden_dim = 128,
            word_dim = 128,
            maxlen=100,
            weight_init_type="ortho_1.0",
            debug=False,
            grad_clip_thresh=1.0,
            ):
        self.adversarial = adversarial
        self.adv_epsilon = adv_epsilon
        self.adv_alpha = adv_alpha

        self.hdim = hidden_dim
        self.wdim = word_dim
        self.maxlen = maxlen
        self.weight_init_type = weight_init_type
        self.grad_clip_thresh =grad_clip_thresh
        # util.madness()

        self.debug = debug

        # Set the random number generators' seeds for consistency
        self.SEED = 123
        # numpy.random.seed(self.SEED)

        self.model_options = None


        # ff: Feed Forward (normal neural net), only useful to put after lstm
        #     before the classifier.
        self.layers = {
              'lstm': (param_init_lstm, lstm_layer),
              'rnn_vanilla': (param_init_rnn_vanilla, rnn_vanilla_layer),
                }
        self.params = None
        self.tparams = None
        self.model_has_been_trained = False


    def zipp(self, params1, params2):
        """
        When we reload the model. Needed for the GPU stuff.
        """
        for kk, vv in params1.iteritems():
            params2[kk].set_value(vv)


    def unzip(self, zipped):
        """
        When we pickle the model. Needed for the GPU stuff.
        """
        new_params = OrderedDict()
        for kk, vv in zipped.iteritems():
            new_params[kk] = vv.get_value()
        return new_params


    def dropout_layer(self, state_before, use_noise, trng):
        proj = tensor.switch(use_noise,
                             (state_before *
                              trng.binomial(state_before.shape,
                                            p=0.5, n=1,
                                            dtype=state_before.dtype)),
                             state_before * 0.5)
        return proj

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
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

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)



    def init_params(self, options):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.
        ----------------------------------------------------------------
        Randomly initializes:
            -the word embedding matrix
                -shape = (n_words,wdim)
            -the weight matrix U
                -shape = (hdim, y_dim)
            -the intercept b
                -shape = (y_dim,)
        """
        self.params = OrderedDict()
        # embedding

        # if 'imdb_lstm' in options['dataset']:
        if options['wemb_init'] =='word2vec':
            # self.params['Wemb'] = load_pretrained_word_embeddings(self.wdim, options['dataset'])
            self.params['Wemb'] = load_pretrained_word_embeddings(self.wdim, 'imdb')
        elif options['wemb_init'] == 'random':
            self.params['Wemb'] = randomly_initialize_word_embeddings(self.wdim, options["n_words"])
        else:
            print "unrecognized word embedding initialization %s.  initializing randomly."%options['wemb_init']



        # embedding ends
        self.params = self.get_layer(options['encoder'])[0](options,
                                                  self.params,
                                                  prefix=options['encoder'],
                                                  init_type=self.weight_init_type
                                                  )

        # classifier
        self.params['U'] = 0.01 * numpy.random.randn(self.hdim,
                                                options['ydim']).astype(config.floatX)
        self.params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)



    def load_params(self, path):
        pp = numpy.load(path)
        for kk, vv in self.params.iteritems():
            if kk not in pp:
                raise Warning('%s is not in the archive' % kk)
            self.params[kk] = pp[kk]

        # return self.params

    def faulty_load_params(self, path):
        """
        assumes that the saved parameters fully specify the model, have these particular names
        """
        self.params = {}
        pp = numpy.load(path)
        for kk in ['Wemb','lstm_W','lstm_U','lstm_b','U','b']:
            self.params[kk] = pp[kk]


    def init_tparams(self):
        self.tparams = OrderedDict()
        for kk, pp in self.params.iteritems():
            self.tparams[kk] = theano.shared(self.params[kk], name=kk)

        # return self.tparams


    def get_layer(self, name):
        # fns = e.g. (param_init_lstm, lstm_layer)
        fns = self.layers[name]
        return fns


    def build_model(self, options):
        """
        #-------------------------------------------------------------
        creates the symbolic variables used by the model.  These include:
            -x: the input matrix, where each word is represented as an index
                -shape (n_timesteps, n_samples)
            -y:
                shape:
            -emb: analogous to x, only using the full embedding of the word.
                -shape (n_timesteps, n_samples, wdim)
            -proj: an lstm_layer instance
        """
        trng = RandomStreams()#self.SEED)

        # Used for dropout.
        use_noise = theano.shared(numpy_floatX(0.))

        x = tensor.matrix('x', dtype='int64')

        mask = tensor.matrix('mask', dtype=config.floatX)
        y = tensor.vector('y', dtype='int64')

        # note that some sequences are padded
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        self.emb = self.tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    self.wdim])

        if self.grad_clip_thresh:
            self.emb = theano.gradient.grad_clip(self.emb, -self.grad_clip_thresh, self.grad_clip_thresh)

        # self.get_layer returns (param_init_lstm, lstm_layer)
        # TODO: why does this not crash when options['encoder'] is not equal to 'lstm'?
        proj = self.get_layer(options['encoder'])[1](self.tparams, self.emb, options,
                                                prefix=options['encoder'],
                                                mask=mask)

        if self.grad_clip_thresh:
            proj = theano.gradient.grad_clip(proj, -self.grad_clip_thresh, self.grad_clip_thresh)

        if options['encoder'] == 'lstm':
            # mean pooling layer
            proj = (proj * mask[:, :, None]).sum(axis=0)
            proj = proj / mask.sum(axis=0)[:, None]
        if options['use_dropout']:
            proj = self.dropout_layer(proj, use_noise, trng)

        pred = tensor.nnet.softmax(tensor.dot(proj, self.tparams['U']) + self.tparams['b'])

        self.f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
        self.f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

        off = 1e-8
        if pred.dtype == 'float16':
            off = 1e-6

        # the objective function:
        self.cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

        # if self.grad_clip_thresh:
        #     self.cost = theano.gradient.grad_clip(self.cost, -self.grad_clip_thresh, self.grad_clip_thresh)

        if self.adversarial:  # done by Isaac
            # adv_x = tensor.matrix('adv_x', dtype='int64')
            # adv_mask = tensor.matrix('adv_mask', dtype=config.floatX)
            leaf_grads = tensor.grad(self.cost, wrt=self.emb) # on all word embeddings

            # treat this as a constant. !!!!!
            # e.g. stop_gradient ("something like this")
            # Victor Zhong
            anti_example = tensor.sgn(leaf_grads)  # word embedding + perturbation
            adv_example = self.emb + self.adv_epsilon*anti_example
            adv_example = theano.gradient.disconnected_grad(adv_example)

            adv_proj = self.get_layer(options['encoder'])[1](self.tparams, adv_example, options,
                                                prefix=options['encoder'],
                                                mask=mask)  # all the edges of LSTM layers (tensor representing all the hidden states)

            if options['encoder'] == 'lstm':  # this is the mean_pooling layer
                adv_proj = (adv_proj * mask[:, :, None]).sum(axis=0)
                adv_proj = adv_proj / mask.sum(axis=0)[:, None]
            if options['use_dropout']:
                adv_proj = self.dropout_layer(adv_proj, use_noise, trng)
            # theano.printing.debugprint(adv_proj)
            # adv_pred = tensor.nnet.softmax(tensor.dot(proj, self.tparams['U']) + self.tparams['b'])
            adv_pred = tensor.nnet.softmax(tensor.dot(adv_proj, self.tparams['U']) + self.tparams['b'])
            # adv_f_pred_prob = theano.function([x, mask], pred, name='adv_f_pred_prob')
            # adv_f_pred_prob = theano.function([x, mask], adv_pred, name='adv_f_pred_prob')

            # adv_f_pred = theano.function([x, mask], adv_pred.argmax(axis=1), name='adv_f_pred')
            adv_cost = -tensor.log(adv_pred[tensor.arange(n_samples), y] + off).mean()

            self.cost = self.adv_alpha*self.cost + (1-self.adv_alpha)*adv_cost
        # theano.printing.pydotprint(cost, outfile="output/lstm_cost_viz.png", var_with_name_simple=True)
        return use_noise, x, mask, y #, f_pred_prob, f_pred, cost


    def pred_probs(self, f_pred_prob, prepare_data, data, iterator, verbose=False):
        """ If you want to use a trained model, this is useful to compute
        the probabilities of new examples.
        """
        n_samples = len(data[0])
        probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

        n_done = 0


        for _, valid_index in iterator:
            x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                      numpy.array(data[1])[valid_index],
                                      maxlen=None)
            pred_probas = f_pred_prob(x, mask)
            probs[valid_index, :] = pred_probas

            n_done += len(valid_index)
            if verbose:
                print '%d/%d samples classified' % (n_done, n_samples)

        return probs


    def pred_error(self, f_pred, prepare_data, data, iterator, verbose=False):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """
        valid_err = 0
        for _, valid_index in iterator:
            x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                      numpy.array(data[1])[valid_index],
                                      maxlen=None)
            preds = f_pred(x, mask)

            targets = numpy.array(data[1])[valid_index]
            valid_err += (preds == targets).sum()
        valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

        return valid_err

    def create_and_save_adversarial_examples(self,
                                            saved_model_fpath,
                                            n_examples=100,
                                            dataset="data/imdb.pkl",
                                            saveto = "output/adversarial_examples.npz",
                                            ):
        """
        recreates the model from saved parameters, then finds adversarial examples.

        right now, not especially modular :(

        Allen's note: n_examples is not used

        :param string model_fname: the name of the file where the model has been stored.
        """



        # below: assert that the training has been done
        assert self.model_has_been_trained

        # we want to have trained nonadversarially in order to have
        # examples that are demonstrative of adversarialness
        assert not self.adversarial


        (_, x_sym, mask_sym, y_sym) =\
             self.build_model(self.model_options,)

        grad_wrt_emb = tensor.grad(self.cost, wrt=self.emb)[0]

        anti_example = tensor.sgn(grad_wrt_emb)

        adv_example = self.emb + self.adv_epsilon*anti_example

        f_adv_example = theano.function([x_sym, mask_sym, y_sym], adv_example, name='f_adv_example')
        f_identity = theano.function([x_sym], self.emb, name='f_identity')


        # 1. get the data
        print 'Loading data'
        #TODO: remove magic 10000!!!
        train, valid, test = load_data(n_words=10000, valid_portion=0.05,
                                       maxlen=self.maxlen, path=dataset)



        corpus = valid
        # make a datastructure in which to store them
        print len(corpus[1])
        sentences_and_adversaries = {
            'original_sentences': None,
            'adversarial_sentences': None,
            'saved_model_fpath' : saved_model_fpath,

            #metadata
            'n_sentences': len(corpus[1]),
            'adversarial_parameters': {
                        'alpha':self.adv_alpha,
                        'epsilon':self.adv_epsilon,
                        },
        }


        x_itf, mask_itf, y_itf = prepare_data(corpus[0], corpus[1])

        # print f_adv_example(x_itf, mask_itf, y_itf)
        # print f_adv_example(x_itf, mask_itf, y_itf).shape

        sentences_and_adversaries['adversarial_sentences'] = f_adv_example(x_itf, mask_itf, y_itf)
        sentences_and_adversaries['original_sentences'] = f_identity(x_itf)

        numpy.savez(saveto, sentences_and_adversaries)#, open(saveto, 'wb'))


    def train_lstm(self,
        saveto, # The best model will be saved there
        dataset,

        #----------------------------------------------------------------------
        #algorithmic hyperparameters
        encoder='lstm',  # TODO: can be removed must be lstm.
        l2_reg_U=0.,  # Weight decay for the classifier applied to the U weights.
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        optimizer="adadelta",  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
        batch_size=16,  # The batch size during training.
        wemb_init='word2vec',

        #----------------------------------------------------------------------
        #parameters related to convergence, saving, and similar
        max_epochs=5000,  # The maximum number of epoch to run
        patience=10,  # Number of epoch to wait before early stop if no progress
        dispFreq=10,  # Display to stdout the training progress every N updates
        n_words=10000,  # Vocabulary size
        validFreq=370,  # Compute the validation error after this number of update.
        saveFreq=1110,  # Save the parameters after every saveFreq updates
        valid_batch_size=64,  # The batch size used for validation/test set.

        #----------------------------------------------------------------------
        # Parameter for extra option (whatever that means)
        noise_std=0.,
        use_dropout=True,  # if False slightly faster, but worst test error
                           # This frequently need a bigger model.
        reload_model=None,  # Path to a saved model we want to start from.
        return_after_reloading=False,  # Path to a saved model we want to start from.
        test_size=-1,  # If >0, we keep only this number of test example.
    ):

        optimizer = OPTIMIZERS[optimizer]
        # Model options
        self.model_options = locals().copy()


        if reload_model:
            self.faulty_load_params(reload_model)
            # self.init_tparams()
            _, self.wdim = self.params['Wemb'].shape
            self.hdim, ydim = self.params['U'].shape

            self.model_options['ydim'] = ydim
            print _, self.wdim, self.hdim, ydim


        self.model_options['hdim'] = self.hdim
        self.model_options['wdim'] = self.wdim

        self.model_options['grad_clip_thresh'] = self.grad_clip_thresh
        print "model options", self.model_options

        # load_data, prepare_data = get_dataset(dataset)

        print 'Loading data'
        #each of the below is a tuple of
        # (list of sentences, where each is a list fo word indices,
        #  list of integer labels)
        if not reload_model:
            train, valid, test =  load_data(n_words=n_words, valid_portion=0.05,
                                           maxlen=self.maxlen, path=dataset)

            if test_size > 0:
                # The test set is sorted by size, but we want to keep random
                # size example.  So we must select a random selection of the
                # examples.
                idx = numpy.arange(len(test[0]))
                numpy.random.shuffle(idx)
                idx = idx[:test_size]
                test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

            ydim = numpy.max(train[1]) + 1

            self.model_options['ydim'] = ydim

        print 'Building model'

        if not reload_model:
            # initialize the word embedding matrix and the parameters of the model (U and b) randomly
            # self.params is a dict mapping name (string) -> numpy ndarray
            self.init_params(self.model_options)

        # This creates Theano Shared Variable from the parameters.
        # Dict name (string) -> Theano Tensor Shared Variable
        # self.params and self.tparams have different copy of the weights.
        self.init_tparams()

        # use_noise is for dropout
        (use_noise, x, mask, y) =\
             self.build_model(self.model_options,)
         # f_pred_prob, self.f_pred, cost)


        if l2_reg_U > 0.:
            l2_reg_U = theano.shared(numpy_floatX(l2_reg_U), name='l2_reg_U')
            weight_decay = 0.
            weight_decay += (self.tparams['U'] ** 2).sum()
            weight_decay *= l2_reg_U
            self.cost += weight_decay

        f_cost = theano.function([x, mask, y], self.cost, name='f_cost')

        grads = tensor.grad(self.cost, wrt=self.tparams.values())
        f_grad = theano.function([x, mask, y], grads, name='f_grad')

        lr = tensor.scalar(name='lr')
        f_grad_shared, f_update = optimizer(lr, self.tparams, grads,
                                            x, mask, y, self.cost)

        if self.debug:
            util.colorprint("Following is the graph of the shared gradient function (f_grad_shared):", "blue")
            theano.printing.debugprint(f_grad_shared.maker.fgraph.outputs[0])

        if return_after_reloading:
            self.model_has_been_trained = True
            return

        print 'Optimization'

        kf_valid = self.get_minibatches_idx(len(valid[0]), valid_batch_size)
        kf_test = self.get_minibatches_idx(len(test[0]), valid_batch_size)

        print "%d train examples" % len(train[0])
        print "%d valid examples" % len(valid[0])
        print "%d test examples" % len(test[0])

        history_errs = []
        best_p = None
        bad_count = 0

        if validFreq == -1:
            validFreq = len(train[0]) / batch_size
        if saveFreq == -1:
            saveFreq = len(train[0]) / batch_size

        uidx = 0  # the number of update done
        estop = False  # early stop
        start_time = time.time()



        try:
            for epoch in xrange(max_epochs):
                sys.stdout.flush()
                n_samples = 0

                # Get new shuffled index for the training set.
                minibatches = self.get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

                for _, train_index_list in minibatches:
                    uidx += 1
                    use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    y = [train[1][t] for t in train_index_list]
                    x = [train[0][t]for t in train_index_list]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, y)
                    n_samples += x.shape[1]

                    cur_cost_val = f_grad_shared(x, mask, y)
                    f_update(lrate)

                    if numpy.isnan(cur_cost_val) or numpy.isinf(cur_cost_val):
                        print 'NaN detected'
                        return 1., 1., 1.

                    if numpy.mod(uidx, dispFreq) == 0:
                        print 'Epoch ', epoch, 'Update ', uidx, 'Cost ', cur_cost_val

                    if saveto and numpy.mod(uidx, saveFreq) == 0:
                        print 'Saving...',

                        if best_p is not None:
                            self.params = best_p
                        else:
                            self.params = self.unzip(self.tparams)
                        numpy.savez(saveto, history_errs=history_errs, **self.params)
                        pkl.dump(self.model_options, open('%s.pkl' % saveto, 'wb'), -1)
                        print 'Done'

                    if numpy.mod(uidx, validFreq) == 0:
                        use_noise.set_value(0.)
                        train_err = self.pred_error(self.f_pred, prepare_data, train, minibatches)
                        valid_err = self.pred_error(self.f_pred, prepare_data, valid,
                                               kf_valid)
                        test_err = self.pred_error(self.f_pred, prepare_data, test, kf_test)

                        history_errs.append([valid_err, test_err])

                        if (uidx == 0 or
                            valid_err <= numpy.array(history_errs)[:,
                                                                   0].min()):

                            best_p = self.unzip(self.tparams)
                            bad_counter = 0

                        print ('Train ', train_err, 'Valid ', valid_err,
                               'Test ', test_err)

                        if (len(history_errs) > patience and
                            valid_err >= numpy.array(history_errs)[:-patience,
                                                                   0].min()):
                            bad_counter += 1
                            if bad_counter > patience:
                                print 'Early Stop!'
                                estop = True
                                break

                print 'Seen %d samples' % n_samples

                if estop:
                    break

        except KeyboardInterrupt:
            print "Training interrupted"

        end_time = time.time()
        if best_p is not None:
            self.zipp(best_p, self.tparams)
        else:
            best_p = self.unzip(self.tparams)

        use_noise.set_value(0.)
        kf_train_sorted = self.get_minibatches_idx(len(train[0]), batch_size)
        train_err = self.pred_error(self.f_pred, prepare_data, train, kf_train_sorted)
        valid_err = self.pred_error(self.f_pred, prepare_data, valid, kf_valid)
        test_err = self.pred_error(self.f_pred, prepare_data, test, kf_test)

        print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
        if saveto:
            numpy.savez(saveto, train_err=train_err,
                        valid_err=valid_err, test_err=test_err,
                        history_errs=history_errs, **best_p)
        print 'The code run for %d epochs, with %f sec/epochs' % (
            (epoch + 1), (end_time - start_time) / (1. * (epoch + 1)))
        print >> sys.stderr, ('Training took %.1fs' %
                              (end_time - start_time))
        self.model_has_been_trained = True
        return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.self,

    ARGS = get_args()
    rnn = Rnn(ARGS.ADVERSARIAL,
            adv_alpha = ARGS.ADVERSARIAL_ALPHA,
            adv_epsilon = ARGS.ADVERSARIAL_EPSILON,
            hidden_dim = ARGS.HIDDEN_DIM,
            word_dim = ARGS.WORD_DIM,
            maxlen = ARGS.MAXLEN,
            weight_init_type = ARGS.WEIGHT_INIT,
            debug=ARGS.DEBUG,
            grad_clip_thresh=ARGS.GRAD_CLIP_THRESH,
            )

    DATASET = "data/%s.pkl"%ARGS.DATANAME

    # adv_decriptor = "adv=%s_"%ARGS.ADVERSARIAL + ("" if not ARGS.ADVERSARIAL else "eps=%.3f_alpha=%.3f"%(ARGS.ADVERSARIAL_EPSILON, ARGS.ADVERSARIAL_EPSILON))
    adv_decriptor = "adv=%s_"%ARGS.ADVERSARIAL + "eps=%.3f_alpha=%.3f"%(ARGS.ADVERSARIAL_EPSILON, ARGS.ADVERSARIAL_EPSILON)
    # a string describing the parameters of this run
    # TODO: change this?
    run_descriptor = 'encoder=%s_%s_data=%s_maxepochs=%s'%(ARGS.ENCODER, adv_decriptor, ARGS.DATANAME, ARGS.MAX_EPOCHS)
    run_descriptor += "_hdim=%s_reg=%.3f_lrate=%.4f_opt=%s_batchsize=%s_clip=%s_weight-init=%s"%(ARGS.HIDDEN_DIM, ARGS.l2_reg_U, ARGS.LRATE, ARGS.OPTIMIZER, ARGS.BATCH_SIZE, ARGS.GRAD_CLIP_THRESH, ARGS.WEIGHT_INIT)

    run_descriptor += "_%s"%util.time_string()
    if ARGS.ID:
        run_descriptor = ARGS.ID + "_" + run_descriptor

    if 1:
        run_descriptor_numeric = str(hash(run_descriptor)) + "_" + util.random_string_signature(4)

        run_descriptor_numeric += "_%s"%util.time_string()
        if ARGS.ID:
            run_descriptor_numeric = ARGS.ID + "_" + run_descriptor_numeric



    MODEL_SAVETO = 'saved_models/%s.npz'%run_descriptor
    RUN_OUTPUT_FNAME = 'output/adv=%s_%s.out'%(ARGS.ADVERSARIAL, run_descriptor_numeric)


    logfile = None
    if ARGS.REDIRECT_OUTPUT_TO_FILE:
        logfile = open(RUN_OUTPUT_FNAME, 'w')
        print util.colorprint("printing all standard output and error to %s...."%RUN_OUTPUT_FNAME, 'rand')
        sys.stdout = logfile
        sys.stderr = logfile

    print "full name of file: ", run_descriptor

    print MODEL_SAVETO

    # print "ARGS.ADVERSARIAL = %s"%ARGS.ADVERSARIAL
    # print "DATASET = %s"%DATASET
    # print "ARGS.MAX_EPOCHS = %s"%ARGS.MAX_EPOCHS
    # print "in summary: \n%s\n"%run_descriptor

    print "ARGS: "
    print ARGS
    print '-'*80
    sys.stdout.flush()

    # RELOAD_MODEL =  "saved_models/lstm__eps=0.5_aplha=0.5_data=imdb_maxepochs=1000_Nov-17-2015.npz"

    rnn.train_lstm(
        dataset = DATASET,
        saveto = MODEL_SAVETO,
        max_epochs = ARGS.MAX_EPOCHS,

        encoder = ARGS.ENCODER, #TODO: pass this into __init__
        l2_reg_U = ARGS.l2_reg_U,
        lrate = ARGS.LRATE,
        optimizer = ARGS.OPTIMIZER,
        batch_size = ARGS.BATCH_SIZE,
        wemb_init = ARGS.WEMB_INIT,
        # reload_model = 'saved_models/lstm_adv=False_data=toy_corpus_maxepochs=2_Nov-17-2015.npz',
        # reload_model = RELOAD_MODEL,
        # return_after_reloading = True,
        test_size=500,
    )
    if ARGS.REDIRECT_OUTPUT_TO_FILE:
        logfile.close()

    # rnn.create_and_save_adversarial_examples(saveto="output/adversarial_examples_%s.npz"%"Allen_success", saved_model_fpath=RELOAD_MODEL)




