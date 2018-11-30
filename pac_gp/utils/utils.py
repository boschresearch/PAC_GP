# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import abc

import numpy as np
import tensorflow as tf


def clamp_and_round(x, x_min_log, x_max_log, digits):
    """
    discretizes x in log-space

    input:
    x           :   value to be discretized
    x_min_log   :   min value
    x_max_log   :   max value
    digits      :   number of digits
    """
    log_x = np.log(x)
    log_x = np.round(log_x, decimals=int(digits))

    log_x[log_x < x_min_log] = x_min_log
    log_x[log_x > x_max_log] = x_max_log

    return np.exp(log_x)


def expand_vector(x, shapes):
    """
    Transforms x into a list of vectors, where the shape of the vectors is
    defined in the list shapes

    inputs:
    x       :     1D-vector
    shapes  :     list of shapes
    """
    shape_lengths = np.array([int(np.prod(shape)) for shape in shapes])
    count = np.sum(shape_lengths)

    assert x.ndim == 1, 'vector should be 1D for expansion (is %dD)' % x.ndim
    assert len(x) == count, 'vector length incompatible with shapes'

    end_pos = np.cumsum(shape_lengths)
    start_pos = np.concatenate(([0], end_pos[:-1]))

    results = []
    for start, end, shape in zip(start_pos, end_pos, shapes):
        results.append(np.reshape(x[start:end], shape))

    return results


def flatten(variables):
    """
    returns a stacked list of variables where each variable has been
    transformed into a 1D-vector

    input:
    variables   :   list of variables
    """
    return np.concatenate([np.reshape(var, (-1, )) for var in variables])


def variable_summaries(var, name=None, vector=True):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope(name, 'summaries', [var]):
        var = tf.convert_to_tensor(var)
        coll = ['stats_summaries']
        if vector:
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean, collections=coll)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev, collections=coll)
            tf.summary.scalar('max', tf.reduce_max(var), collections=coll)
            tf.summary.scalar('min', tf.reduce_min(var), collections=coll)
            tf.summary.histogram('histogram', var, collections=coll)
        else:
            tf.summary.scalar('mean', tf.reduce_mean(var), collections=coll)


class Configurable(object):
    """
    abstract base class for setting configurations
    """
    __metaclass__ = abc.ABCMeta

    def configure(self, config):
        error_msg = 'configure method of %s expects dict type config parameter'
        assert type(config) == dict, error_msg % (self.__class__)

        # Copy all attributes from config-dict to the class's local space
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise NotImplementedError('Unknown attribute %s for %s' %
                                          (key, self.name))


def vec_to_tri(vectors, N):
    """
    Takes a D x M tensor `vectors' and maps it to a
    D x matrix_size x matrix_size tensor
    where the where the lower triangle of each matrix_size x matrix_size matrix
    is constructed by unpacking each M-vector.

    Native TensorFlow version of Custom Op by Mark van der Wilk.

    def int_shape(x):
        return list(map(int, x.get_shape()))

    D, M = int_shape(vectors)
    N = int( np.floor( 0.5 * np.sqrt( M * 8. + 1. ) - 0.5 ) )
    # Check M is a valid triangle number
    assert((matrix * (N + 1)) == (2 * M))
    """
    indices = list(zip(*np.tril_indices(N)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

    def vec_to_tri_vector(vector):
        return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

    return tf.map_fn(vec_to_tri_vector, vectors)
