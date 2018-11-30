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

from pac_gp.utils.utils_tf import py_func


def BinaryKLInv(x, eps, name=None):
    with tf.name_scope(name, "BinaryKLInv", [x, eps]) as name:
        y = py_func(binary_kl_inv, [x, eps], tf.float64, name=name,
                    grad=BinaryKLInvGrad)
        return y


def BinaryKLInvGrad(op, grad):
    """The gradients for 'binary_kl_inv'.

    Args:
            we can use to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the 'binary_kl_inv' op.

    Returns:
        Gradients with respect to the input of 'binary_kl_inv'.
    """
    risk = op.inputs[0]
    pen = op.inputs[1]
    dx, deps = tf.py_func(binary_kl_inv_grad,
                          [risk, pen],
                          [tf.float64, tf.float64])
    return dx * grad, deps * grad


def binary_kl_inv(x, eps, acc=1e-7):
    """
    Args:
        x, in [0,1] (if x<0, then interpret as 0; if x>1, then interpret as 1)
        eps, in [0,Inf] (if eps<0, then interpret as 0)
        acc, (optional): accuracy goal (on returned value of y)
    Returns:
        y in the interval [x,1], such that KLbin(x||y)=eps.
        (eps=Inf produces y=1)
    """
    acc = 1e-7

    # X(p) <=0 : q = 1- np.exp(-eps)
    if x <= 0:
        y = 1. - np.exp(-eps)
        return y
    if x >= 1.0:
        y = np.exp(-eps)
        return y

    if np.isinf(eps):
        y = 1.0
        return y
    if eps <= 0.0:
        y = x
        return y

    xlower = x
    xupper = 1.0

    while (xupper-xlower) > acc:
        if binary_kl(x, (xlower + xupper)/2.0) > eps:
            xupper = (xlower + xupper)/2.0
        else:
            xlower = (xlower + xupper)/2.0

    y = (xlower + xupper)/2.0

    return y


def binary_kl_inv_grad(x, eps):
    if x >= 1.0:
        dx = 1.0
        deps = 0.0
        return [dx, deps]

    if np.isinf(eps):
        dx = 0.0
        deps = 1.0
        return [dx, deps]

    if eps <= 0.0:
        dx = 1.0
        deps = 0.0
        return [dx, deps]

    if x <= 0.0:
        x = 0.0

    y = binary_kl_inv(x, eps)

    if y <= 1e-8:
        y = 1e-8

    if y >= 1.0 - 1e-8:
        y = 1.0 - 1e-8

    if x <= 1e-8:
        x = 1e-8

    if x >= 1.0 - 1e-8:
        x = 1.0 - 1e-8

    deps = y * (1. - y) / (y - x)
    dx = deps * (np.log(1.0 - x) + np.log(y) - np.log(x) - np.log(1.0 - y))

    return dx, deps


def binary_kl(p, q):

    p = np.asarray(p)
    q = np.asarray(q)

    p, q = np.broadcast_arrays(p, q)

    kl = np.zeros_like(p, dtype=np.float64)

    # p <= 0.0
    kl[(p <= 0.0) & (q <= 0.0)] = 0.0
    kl[(p <= 0.0) & (q >= 1.0)] = np.inf
    ind = (p <= 0.0) & (q > 0.0) & (q < 1.0)
    kl[ind] = -np.log(1.0 - q[ind])

    # p >= 1.0
    kl[(p >= 1.0) & (q <= 0.0)] = np.inf
    kl[(p >= 1.0) & (q >= 1.0)] = 0.0
    ind = (p >= 1.0) & (q > 0.0) & (q < 1.0)
    kl[ind] = -np.log(q[ind])

    # 0.0 < p < 1.0
    kl[(p > 0.0) & (p < 1.0) & (q <= 0.0)] = np.inf
    kl[(p > 0.0) & (p < 1.0) & (q >= 1.0)] = np.inf
    ind = (p > 0.0) & (p < 1.0) & (q > 0.0) & (q < 1.0)
    kl[ind] = p[ind] * (np.log(p[ind]) - np.log(q[ind]))
    kl[ind] += (1.0 - p[ind]) * (np.log(1.0 - p[ind]) - np.log(1.0 - q[ind]))

    return kl
