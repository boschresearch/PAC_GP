# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import numpy as np
from scipy.stats import norm


def mean_squared_error(Y_true, Ymean_hat, **kwargs):
    """
    evaluates mean squared error

    input:
    Y_true      :   true outcome
    Ymean_hat   :   predicted mean
    """
    assert Y_true.shape == Ymean_hat.shape
    return np.mean((Y_true-Ymean_hat)**2)


def epsilon_loss(Y_true, Ymean_hat, epsilon, **kwargs):
    """
    evaluates 0-1 loss

    input:
    Y_true      :   true outcome
    Ymean_hat   :   predicted mean
    epsilon     :   sensitivity parameter of the loss function
    """
    return np.mean(np.absolute(Y_true-Ymean_hat) > epsilon)


def inv_gauss(Y_true, Ymean_hat, Yvar_hat, epsilon, fmean_hat, fvar_hat,
              **kwargs):
    """
    empirical risk associated with the loss function
    1 - exp(-((y-y_hat)/epsilon)^2)

    inputs:
    Y_true      :   true outcome
    Ymean_hat   :   predict mean
    fvar_hat    :   predicted variance (noise-less)
    epsilon     :   sensitivity parameter of the loss function
    """
    sf = np.exp(-(Y_true-Ymean_hat)**2/(2*fvar_hat + epsilon**2))
    inv_gauss = 1./np.sqrt(1 + 2*fvar_hat/epsilon**2) * sf
    return 1-np.mean(inv_gauss)


def neg_ll(Y_true, Ymean_hat, Yvar_hat, **kwargs):
    """
    returns negative log-likelihood of the test data

    inputs:
    Y_true      :   true outcome
    Ymean_hat   :   predicted mean
    Yvar_hat    :   predicted variance (included noise)
    """
    logdet = 0.5 * np.log(2 * np.pi * Yvar_hat).sum()
    sf = 0.5 * np.sum((Y_true - Ymean_hat)**2 / Yvar_hat)
    return logdet + sf


def bayes_risk(Y_true, Ymean_hat, Yvar_hat, epsilon, **kwargs):
    """
    returns bayes risk w.r.t. 0-1 loss

    inputs:
    Y_true      :   true outcome
    Ymean_hat   :   predicted mean
    Yvar_hat    :   predicted variance (included noise )
    epsilon     :   sensitivity of the loss function
    """
    return epsilon_loss(Y_true, Ymean_hat=Ymean_hat, epsilon=epsilon)


def gibbs_risk_noiseless(Y_true, Ymean_hat, Yvar_hat, epsilon,
                         fmean_hat, fvar_hat, **kwargs):
    """
    returns gibb's risk

    inputs:
    Y_true      :   true outcome
    Ymean_hat   :   predicted mean
    fvar_hat    :   predicted variance (without) noise)
    epsilon     :   sensitivity of the loss function
    """
    return empirical_risk(Y_true=Y_true, Ymean_hat=fmean_hat,
                          Yvar_hat=fvar_hat, epsilon=epsilon)


def empirical_risk(Y_true, Ymean_hat, Yvar_hat, epsilon, **kwargs):
    """
    returns empirical loss associated with the 0-1 loss given the predictive
    distribution N(Y_hat| Ymean_hat, Yvar_hat )

    inputs:
    Y_true      :   true outcome
    Ymean_hat   :   predicted mean
    fvar_hat    :   predicted variance (without noise)
    epsilon     :   sensitivity of the loss function
    """

    cdf_y_p_eps = norm.cdf(Y_true + epsilon, loc=Ymean_hat,
                           scale=np.sqrt(Yvar_hat))
    cdf_y_m_eps = norm.cdf(Y_true - epsilon, loc=Ymean_hat,
                           scale=np.sqrt(Yvar_hat))

    return 1.0 - np.mean(cdf_y_p_eps - cdf_y_m_eps)
