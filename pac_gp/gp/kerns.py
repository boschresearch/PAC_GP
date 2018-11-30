# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""
import numpy as np
import tensorflow as tf
from pac_gp.utils.transformations import Log1pe

"""
The following snippets are derived from GPFlow V 1.0
  (https://github.com/GPflow/GPflow)
Copyright 2017 st--, Mark van der Wilk, licensed under the Apache
License, Version 2.0, cf. 3rd-party-licenses.txt file in the root directory
of this source tree.
"""


class RBF:
    def __init__(self, input_dim, variance=1.0, lengthscale=None, ARD=True):
        with tf.name_scope('kern'):

            self.variance = np.atleast_1d(variance)

            if lengthscale is not None:
                # If lengthscale is given check:
                # 1) individual lengthscale for each dimension
                # or 2) one lengthscale for all dimensions
                lengthscale = np.asarray(lengthscale, dtype=np.float64)
                lengthscale = np.atleast_1d(lengthscale)

                assert_msg = 'Bad number of lengthscale dimensions'
                assert lengthscale.ndim == 1, assert_msg
                assert_msg = 'Bad number of lengthscales'
                assert lengthscale.size in [1, input_dim], assert_msg

                if ARD is True and lengthscale.size != input_dim:
                    lengthscale = np.ones(input_dim)*lengthscale

            else:
                # Default lengthscale if nothing is given
                if ARD is True:
                    # Independent lengthscalea for each dimension
                    lengthscale = np.ones(input_dim)
                    shape = (input_dim, )
                else:
                    # One lengthscale for all dimensions
                    lengthscale = np.ones(1)
                    shape = (1,)

            self.variance_unc_tf = tf.placeholder(dtype=tf.float64,
                                                  shape=(1, ),
                                                  name='variance_unc')

            lengthscales_unc_tf = tf.placeholder(dtype=tf.float64,
                                                 shape=shape,
                                                 name='lengthscales_unc')

        self.lengthscales_unc_tf = lengthscales_unc_tf
        self.lengthscale = lengthscale
        self.input_dim = input_dim
        self.trans = Log1pe()
        self.variance_tf = self.trans.forward_tensor(self.variance_unc_tf)
        self.lengthscales_tf = self.trans.forward_tensor(lengthscales_unc_tf)
        self.ARD = ARD

    def square_dist(self, X, X2):
        X = X / self.lengthscales_tf
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales_tf
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X):
        N = tf.shape(X)[0]
        return tf.fill(tf.stack([N]), tf.squeeze(self.variance_tf))

    def K(self, X, X2=None):
        return self.variance_tf * tf.exp(-self.square_dist(X, X2) / 2)
