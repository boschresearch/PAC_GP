# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import abc
import tensorflow as tf
import numpy as np

from pac_gp.utils.utils import vec_to_tri

"""
The following snippets are derived from GPFlow V 1.0
  (https://github.com/GPflow/GPflow)
Copyright 2017 st--, Mark van der Wilk, licensed under the Apache
License, Version 2.0, cf. 3rd-party-licenses.txt file in the root directory
of this source tree.
"""


class ITransform(metaclass=abc.ABCMeta):
    """
    x is the unconstrained, free-space parameter, which can take any value.
    y is the constrained, "real units" parameter corresponding to the data.

    y = forward(x)
    x = backward(y)
    """

    @abc.abstractmethod
    def forward(self, x):
        """
        Map from the free-space to the variable space, using numpy
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backward(self, y):
        """
        Map from the variable-space to the free space, using numpy
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_tensor(self, x):
        """
        Map from the free-space to the variable space, using tensorflow
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def log_jacobian_tensor(self, x):
        """
        Return the log Jacobian of the forward_tensor mapping.

        Note that we *could* do this using a tf manipulation of
        self.forward_tensor, but tensorflow may have difficulty: it doesn't
        have a Jacobian at time of writing. We do this in the tests to make
        sure the implementation is correct.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self):
        """
        A short string describing the nature of the constraint
        """
        raise NotImplementedError


class Transform(ITransform):  # pylint: disable=W0223
    def __call__(self, other_transform):
        """
        Calling a Transform with another Transform results in a Chain of
        both. The following are equivalent:
        >>> Chain(t1, t2)
        >>> t1(t2)
        """
        if not isinstance(other_transform, Transform):
            msg = "transforms can only be chained with other transforms: "
            msg += "perhaps you want t.forward(x)"
            raise TypeError(msg)
        return Chain(self, other_transform)


class Chain(Transform):
    """
    Chain two transformations together:
    .. math::
       y = t_1(t_2(x))
    where y is the natural parameter and x is the free state
    """

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def forward_tensor(self, x):
        return self.t1.forward_tensor(self.t2.forward_tensor(x))

    def forward(self, x):
        return self.t1.forward(self.t2.forward(x))

    def backward(self, y):
        return self.t2.backward(self.t1.backward(y))

    def log_jacobian_tensor(self, x):
        return self.t1.log_jacobian_tensor(self.t2.forward_tensor(x)) +\
               self.t2.log_jacobian_tensor(x)

    def __str__(self):
        return "{} {}".format(self.t1.__str__(), self.t2.__str__())


class Log1pe(Transform):
    """
    A transform of the form
    .. math::

       y = \log(1 + \exp(x))

    x is a free variable, y is always positive.

    This function is known as 'softplus' in tensorflow.
    """

    def __init__(self, lower=1e-6):
        """
        lower is a float that defines the minimum value that this transform can
        take, default 1e-6. This helps stability during optimization, because
        aggressive optimizers can take overly-long steps which lead to zero in
        the transformed variable, causing an error.
        """
        self._lower = lower

    def forward(self, x):
        """
        Implementation of softplus. Overflow avoided by use of the logaddexp
        function.
        self._lower is added before returning.
        """
        return np.logaddexp(0, x) + self._lower

    def forward_tensor(self, x):
        return tf.nn.softplus(x) + self._lower

    def log_jacobian_tensor(self, x):
        return tf.negative(tf.reduce_sum(tf.nn.softplus(tf.negative(x))))

    def backward(self, y):
        """
        Inverse of the softplus transform:
        .. math::

           x = \log( \exp(y) - 1)

        The bound for the input y is [self._lower. inf[, self._lower is
        subtracted prior to any calculations. The implementation avoids
        overflow explicitly by applying the log sum exp trick:
        .. math::

           \log ( \exp(y) - \exp(0)) &= ys + \log( \exp(y-ys) - \exp(-ys)) \\
                                     &= ys + \log( 1 - \exp(-ys)

           ys = \max(0, y)

        As y can not be negative, ys could be replaced with y itself.
        However, in case :math:`y=0` this results in np.log(0). Hence the zero
        is replaced by a machine epsilon.
        .. math::

           ys = \max( \epsilon, y)


        """
        ys = np.maximum(y - self._lower, np.finfo(np.float64).eps)
        return ys + np.log(-np.expm1(-ys))

    def __str__(self):
        return '+ve'


class LowerTriangular(Transform):
    """
    A transform of the form

       tri_mat = vec_to_tri(x)

    x is a free variable, y is always a list of lower triangular matrices sized
    (N x N x D).
    """

    def __init__(self, N, num_matrices=1, squeeze=False):
        """
        Create an instance of LowerTriangular transform.
        Args:
            N the size of the final lower triangular matrices.
            num_matrices: Number of matrices to be stored.
            squeeze: If num_matrices == 1, drop the redundant axis.
        """
        # We need to store this for reconstruction.
        self.num_matrices = num_matrices
        self.squeeze = squeeze
        self.N = N

    def _validate_vector_length(self, length):
        """
        Check whether the vector length is consistent with being a triangular
         matrix and with `self.num_matrices`.
        Args:
            length: Length of the free state vector.

        Returns: Length of the vector with the lower triangular elements.

        """
        L = length / self.num_matrices
        if int(((L * 8) + 1) ** 0.5) ** 2.0 != (L * 8 + 1):
            raise ValueError("The free state must be a triangle number.")
        return L

    def forward(self, x):
        """
        Transforms from the free state to the variable.
        Args:
            x: Free state vector. Must have length of `self.num_matrices` *
                triangular_number.

        Returns:
            Reconstructed variable.
        """
        L = self._validate_vector_length(len(x))
        matsize = int((L * 8 + 1) ** 0.5 * 0.5 - 0.5)
        xr = np.reshape(x, (self.num_matrices, -1))
        var = np.zeros((matsize, matsize, self.num_matrices), np.float64)
        for i in range(self.num_matrices):
            indices = np.tril_indices(matsize, 0)
            indices += (np.zeros(len(indices[0])).astype(int) + i,)
            var[indices] = xr[i, :]
        return var.squeeze() if self.squeeze else var

    def backward(self, y):
        """
        Transforms from the variable to the free state.
        Args:
            y: Variable representation.

        Returns:
            Free state.
        """
        N = int(np.sqrt(y.size / self.num_matrices))
        reshaped = np.reshape(y, (N, N, self.num_matrices))
        size = len(reshaped)
        triangular = reshaped[np.tril_indices(size, 0)].T
        return triangular

    def forward_tensor(self, x):
        reshaped = tf.reshape(x, (self.num_matrices, -1))
        fwd = tf.transpose(vec_to_tri(reshaped, self.N), [1, 2, 0])
        return tf.squeeze(fwd) if self.squeeze else fwd

    def log_jacobian_tensor(self, x):
        return tf.zeros((1,), np.float64)

    def __str__(self):
        return "LoTri->vec"
