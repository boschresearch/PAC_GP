# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import tensorflow as tf

from pac_gp.gp.mean_functions import Zero
from pac_gp.gp.conditionals import feature_conditional

"""
The following snippets are derived from GPFlow V 1.0
  (https://github.com/GPflow/GPflow)
Copyright 2017 st--, Mark van der Wilk, licensed under the Apache
License, Version 2.0, cf. 3rd-party-licenses.txt file in the root directory
of this source tree.
"""


class GPR:
    """
    Gaussian Process Regression.

    Implementation of full GP regression following GPflow implementation
    """
    def __init__(self, X, Y, sn2, kern, mean_function=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        self.X = X
        self.Y = Y

        self.sn2 = sn2

        self.kern = kern
        self.mean_function = mean_function or Zero()

        self.N = tf.shape(X)[0]   # Number of data points
        self.D = tf.shape(X)[1]   # Input dimensionality
        self.R = tf.shape(Y)[1]   # Output dimensionality
        self.jitter = 1e-06

    def _build_predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        N = tf.shape(self.X)[0]

        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X)
        K += tf.eye(N, dtype=tf.float64) * (self.sn2 + self.jitter)
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar

    def _build_predict_y(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the observations at some new points
        Xnew.
        """
        mean, var = self._build_predict_f(Xnew, full_cov)

        if full_cov is True:
            noise = self.sn2 * tf.eye(tf.shape(Xnew)[0], dtype=tf.float64)
            var = var + noise[:, :, None]
        else:
            var = var + self.sn2

        return mean, var


class GPRFITC:
    def __init__(self, X, Y, sn2, kern, mean_function=None, Z=None):
        self.X = X
        self.Y = Y
        self.Z = Z

        self.sn2 = sn2

        self.kern = kern
        self.mean_function = mean_function or Zero()

        self.N = tf.shape(X)[0]   # Number of data points
        self.D = tf.shape(X)[1]   # Input dimensionality
        self.M = tf.shape(Z)[0]   # Number of inducing points
        self.R = tf.shape(Y)[1]   # Output dimensionality

    def _build_common_terms(self):
        err = self.Y - self.mean_function(self.X)  # size N x R
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.kern.K(self.Z, self.X)
        Kuu = self.kern.K(self.Z) + 1e-6 * tf.eye(self.M, dtype=tf.float64)

        # choelsky: Luu Luu^T = Kuu
        Luu = tf.cholesky(Kuu)
        #  V^T V = Qff = Kuf^T Kuu^-1 Kuf
        V = tf.matrix_triangular_solve(Luu, Kuf)

        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + self.sn2

        B = tf.eye(self.M, dtype=tf.float64)
        B += tf.matmul(V / nu, V, transpose_b=True)
        L = tf.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size N x R
        alpha = tf.matmul(V, beta)  # size N x R

        gamma = tf.matrix_triangular_solve(L, alpha, lower=True)  # size N x R

        return err, nu, Luu, L, alpha, beta, gamma

    def _build_predict_f(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        _, _, Luu, L, _, _, gamma = self._build_common_terms()
        Kus = self.kern.K(self.Z, Xnew)  # size  M x Xnew

        w = tf.matrix_triangular_solve(Luu, Kus, lower=True)  # size M x Xnew

        tmp = tf.matrix_triangular_solve(tf.transpose(L), gamma, lower=False)
        mean = tf.matmul(w, tmp, transpose_a=True) + self.mean_function(Xnew)
        intermediateA = tf.matrix_triangular_solve(L, w, lower=True)

        if full_cov:
            var = self.kern.K(Xnew) - tf.matmul(w, w, transpose_a=True) \
                  + tf.matmul(intermediateA, intermediateA, transpose_a=True)
            var = tf.tile(tf.expand_dims(var, 2), tf.stack([1, 1, self.R]))
        else:
            var = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(w), 0) \
                  + tf.reduce_sum(tf.square(intermediateA), 0)  # size Xnew,
            var = tf.tile(tf.expand_dims(var, 1), tf.stack([1, self.R]))

        return mean, var

    def _build_predict_y(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the observations at some new points
        Xnew.
        """
        mean, var = self._build_predict_f(Xnew, full_cov)

        if full_cov is True:
            noise = self.sn2 * tf.eye(tf.shape(Xnew)[0], dtype=tf.float64)
            var = var + noise[:, :, None]
        else:
            var = var + self.sn2

        return mean, var


class SVGP:
    """
    This is the Sparse Variational GP (SVGP).
    Implementation from GPflow.models.svgp.SVGP
    """
    def __init__(self, X, Y, sn2, kern, mean_function, Z,
                 q_mu, q_sqrt, q_diag=True):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kern, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        """
        self.X = X
        self.Y = Y

        # Inducing inputs
        self.Z = Z
        # Inducing outputs
        self.q_mu = q_mu
        self.q_sqrt = q_sqrt

        self.sn2 = sn2

        self.kern = kern
        self.mean_function = mean_function or Zero()

        # Number of data points self.num_data = X.shape[0]
        self.N = tf.shape(X)[0]
        # Input dimensionality
        self.D = tf.shape(X)[1]
        # Number of inducing points
        self.M = tf.shape(Z)[0]
        # Output dimensionality  self.num_latent = Y.shape[1]
        self.R = tf.shape(Y)[1]

        self.q_diag = q_diag

    def _build_predict_f(self, Xnew, full_cov=False):
        mu, var = feature_conditional(Xnew, self.Z, self.kern, self.q_mu,
                                      full_cov=full_cov, q_sqrt=self.q_sqrt)
        return mu + self.mean_function(Xnew), var

    def _build_predict_y(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the observations at some new points
        Xnew.
        """
        mean, var = self._build_predict_f(Xnew, full_cov)

        if full_cov is True:
            noise = self.sn2 * tf.eye(tf.shape(Xnew)[0], dtype=tf.float64)
            var = var + noise[:, :, None]
        else:
            var = var + self.sn2

        return mean, var
