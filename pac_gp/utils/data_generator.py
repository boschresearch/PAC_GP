# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import numpy as np


def generate_sin_data(N=100, x_min=-3, x_max=3,
                      noise_var=0.1**2,
                      random_order=True):

    # Noise free target function
    def f_latent(x):
        x = np.asarray(x)
        return np.sin(x)

    # Target function f_latent + Gaussian noise
    def f(x):
        x = np.asarray(x)
        eps = np.random.normal(loc=0.0, scale=np.sqrt(noise_var),
                               size=x.shape)
        return f_latent(x) + eps

    if random_order:
        X = np.random.uniform(low=x_min, high=x_max, size=(N, 1))
    else:
        X = np.linspace(start=x_min, stop=x_max, num=N)[:, None]

    if noise_var is None:
        return X, f_latent(X)
    else:
        return X, f(X)
