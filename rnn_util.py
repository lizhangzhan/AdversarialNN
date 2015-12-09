# rnn_util.py

# from collections import OrderedDict
# import cPickle as pkl
# import sys
# import time
# import util

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle

def ortho_init(idim, jdim, scale):
    scale = float(scale)  # calling function may pass in a string
    if idim > jdim:
        print "warning: impossible to create full orthonormal (row) basis for " +\
                "overdetermined matrix (shape: (%s, %s)).  Approximating...." % (idim, jdim)
    larger_dim = idim if idim > jdim else jdim
    W = numpy.random.randn(larger_dim, larger_dim)
    u, s, v = numpy.linalg.svd(W)
    basis = scale * u[0:idim, 0:jdim]
    return basis.astype(config.floatX)

def normal_init(idim, jdim, stddev):
    stddev = float(stddev)  # calling function may pass in a string
    W = numpy.random.randn(idim, jdim)
    W *= stddev
    return W.astype(config.floatX)


def _p(pp, name):
    # add a prefix to a name
    return '%s_%s' % (pp, name)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def _slice(_x, n, dim):
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

#=======================================================================


def param_init_lstm(options, params, prefix='lstm', init_type="ortho_1.0"):
    """
    Init the LSTM parameters.

    #------------------------------------------------------------------
    Note that we stack four matrices for computational speedups, so we
    can do all the LSTm updates with one matric multiplication

    :param str init_type: may be "ortho_X.Y" or "normal_X.Y" (e.g. "normal_0.001")
    :see: init_params
    """

    weight_init_options = init_type.split("_")[1:]
    weight_init_type = init_type.split("_")[0]
    weight_init = {
        "ortho": ortho_init,
        "normal": normal_init,
    }[weight_init_type]

    hdim = options['hdim']
    wdim = options['wdim']

    W = numpy.concatenate([weight_init(wdim, hdim, *weight_init_options),
                           weight_init(wdim, hdim, *weight_init_options),
                           weight_init(wdim, hdim, *weight_init_options),
                           weight_init(wdim, hdim, *weight_init_options)], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([weight_init(hdim, hdim, *weight_init_options),
                           weight_init(hdim, hdim, *weight_init_options),
                           weight_init(hdim, hdim, *weight_init_options),
                           weight_init(hdim, hdim, *weight_init_options)], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * hdim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    """
    ------------------------------------------------------------------
    state_below: the word embedding matrix (in the vanilla lstm case)

    :return: a theano symbolic variable representing the lstm layer on
        top of whatever came below it

    variables internal to this function:
        nsteps: the number of timesteps.
            TODO: how is this the same for all input??
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _step(m_, x_, h_, c_):
        """
        ----------------------------------------------------------------------
        For some timestep t (_step is agnostic to the index of the timestep):

        m_ == mask[t, :]  mask for THIS timestep.
        x_ == state_below[t, : (,:)] input from layer below for THIS timestep
                NOTE: the transformation by the matrix W has already be performed.
        h_ == the activation from the PRIOR TIMESTEP (TODO: unknown initialization)
        c_ == c from the PRIOR TIMESTEP (dunno what this is, prolly some lstm thing)
        """
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['hdim']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['hdim']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['hdim']))
        c = tensor.tanh(_slice(preact, 3, options['hdim']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        if options['grad_clip_thresh']:
            h = theano.gradient.grad_clip(
                h, -options['grad_clip_thresh'], options['grad_clip_thresh'])
            c = theano.gradient.grad_clip(
                c, -options['grad_clip_thresh'], options['grad_clip_thresh'])

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    hdim = options['hdim']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           hdim),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           hdim)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)

    #return all hidden activations
    return rval[0]

def param_init_rnn_vanilla(options, params, prefix='rnn_vanilla'):
    """
    Init the rnn_vanilla parameter:

    :see: init_params
    """

    # TODO: I may have reversed the wdim and the hdim
    W = ortho_init(options['wdim'], options['hdim'])
    params[_p(prefix, 'W')] = W
    U = ortho_init(options['hdim'], options['hdim'])
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((options['hdim'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def rnn_vanilla_layer(tparams, state_below, options, prefix='rnn_vanilla', mask=None):
    """
    ------------------------------------------------------------------
    state_below: the word embedding matrix (in the vanilla rnn_vanilla case)

    :return: a theano symbolic variable representing the rnn_vanilla layer on
        top of whatever came below it

    variables internal to this function:
        nsteps: the number of timesteps.
            TODO: how is this the same for all input??
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _step(m_, x_, h_):
        """
        ----------------------------------------------------------------------
        For some timestep t (_step is agnostic to the index of the timestep):

        m_ == mask[t, :]  mask for THIS timestep.
        x_ == state_below[t, : (,:)] input from layer below for THIS timestep
                NOTE: the transformation by the matrix W has already be performed.
        h_ == the activation from the PRIOR TIMESTEP (TODO: unknown initialization)
        c_ == c from the PRIOR TIMESTEP (dunno what this is, prolly some lstm thing)
        """
        preact_from_prev_state = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact_from_cur_word = x_  # tensor.dot(x_, tparams[_p(prefix, 'W')])
        preact = preact_from_prev_state + preact_from_cur_word

        h = tensor.tanh(_slice(preact, 3, options['hdim']))
        # h = tensor.tanh(preact)

        # apply dropout
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    hdim = options['hdim']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           hdim),
                                              # tensor.alloc(numpy_floatX(0.),
                                              #              n_samples,
                                              #              hdim)
                                              ],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)

    return rval[0]


def load_pretrained_word_embeddings(wdim, dataset_path):
    """
    Require options to get the path to dataset
    """
    assert wdim == 300
    idx_f = "data/" + dataset_path.split('.')[0] + ".idxmap.pkl"

    # we load word vectors here
    with open(idx_f) as f:
        fi = cPickle.load(f)
        word_emb, word_idx_map = fi
    # word_emb should be the exact order of how we index vocab
    # so no additional steps are required.
    return word_emb.astype(config.floatX)


def randomly_initialize_word_embeddings(wdim, n_words, scale=0.01):
    randn = numpy.random.rand(n_words, wdim)
    return (scale * randn).astype(config.floatX)

#=========================================================================
# OPTIMIZERS


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd than needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update
