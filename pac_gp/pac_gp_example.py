# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import GPy

import numpy as np
import matplotlib.pyplot as plt

from pac_gp.gp.mean_functions import Zero
from pac_gp.gp.kerns import RBF
from pac_gp.gp.pac_gp import PAC_INDUCING_HYP_GP

from pac_gp.utils.data_generator import generate_sin_data

# %% Configuration

# Number of data points
N_train = 50
N_test = 100

# Number of inducing inputs
M = 10

# Input space dimension
D = 1

x_min = -3
x_max = 3
dx = (x_max - x_min) / 6.0

epsilon_np = 0.1
delta_np = 0.001

# %% Generate data

x_data, y_data = generate_sin_data(N_train, x_min+dx, x_max-dx,
                                   0.1**2, random_order=True)
x_true, y_true = generate_sin_data(N_test, x_min, x_max, None,
                                   random_order=False)

# %% Set up and train GPy model for comparison

kernel = GPy.kern.RBF(input_dim=D, ARD=True)

full_gpy = GPy.models.GPRegression(x_data, y_data, kernel=kernel)
full_gpy.optimize()

sparse_gpy = GPy.models.SparseGPRegression(x_data, y_data,
                                           kernel=kernel, num_inducing=M)
sparse_gpy.optimize()

# Initialize GP parameters from optimized sparse GP model (GPy)
sf2_gpy = sparse_gpy.rbf.variance.values
sn2_gpy = sparse_gpy.Gaussian_noise.variance.values
lengthscales_gpy = sparse_gpy.rbf.lengthscale.values
z_gpy = sparse_gpy.inducing_inputs.values

# %% Set up and train PAC-GP model

kern = RBF(D)
mean = Zero()
pac_gp = PAC_INDUCING_HYP_GP(X=x_data, Y=y_data, Z=z_gpy,
                             sn2=sn2_gpy,
                             kernel=kern, mean_function=mean,
                             epsilon=epsilon_np, delta=delta_np,
                             verbosity=0,
                             method='bkl', loss='01_loss')
pac_gp.optimize()
Z_opt = pac_gp.Z

# %% Predict on test data

y_mean_full_gpy, y_var_full_gpy = full_gpy.predict(x_true)
y_mean_sparse_gpy, y_var_sparse_gpy = sparse_gpy.predict(x_true)
y_mean_pac_gp, y_var_pac_gp = pac_gp.predict(Xnew=x_true, full_cov=False)

# %% Plot data and GPy/GP tf predictions including PAC GP

plt.figure('Data and GPy/GPtf/PAC GP predictions')
plt.clf()

plt.subplot(1, 3, 1)
plt.title('Full GP (GPy)')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_full_gpy)
error = np.squeeze(2 * np.sqrt(y_var_full_gpy))
plt.plot(x_true, y, '-', color='C2', label='full GP (GPy)')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)
plt.xlabel('input $x$')
plt.ylabel('output $y$')
plt.grid()
plt.legend(loc=2)
plt.ylim([-1.5, 1.5])
plt.xlim([-3, 3])

plt.subplot(1, 3, 2)
plt.title('Sparse GP (GPy)')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_sparse_gpy)
error = np.squeeze(2 * np.sqrt(y_var_sparse_gpy))
plt.plot(x_true, y, '-', color='C2', label='sparse GP (GPy)')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)

plt.plot(np.squeeze(sparse_gpy.inducing_inputs), -1.5 * np.ones((M, )), 'o',
         color='C3', label='GPy inducing inputs')
plt.xlabel('input $x$')
plt.ylabel('output $y$')
plt.grid()
plt.legend(loc=2)
plt.ylim([-1.5, 1.5])
plt.xlim([-3, 3])

plt.subplot(1, 3, 3)
plt.title('PAC GP')
plt.plot(x_data, y_data, '+', label='data points')
plt.plot(x_true, y_true, '-', label='true function')

y = np.squeeze(y_mean_pac_gp)
error = np.squeeze(2 * np.sqrt(y_var_pac_gp))
plt.plot(x_true, y, '-', color='C2', label='sparse GP (tf)')
plt.fill_between(np.squeeze(x_true), y-error, y+error, color='C2', alpha=0.3)

plt.plot(np.squeeze(z_gpy), -1.5 * np.ones((M, )), 'o',
         color='C3', label='GPy inducing inputs')
plt.plot(np.squeeze(Z_opt), -1.5 * np.ones((M, )), 'o',
         color='C4', label='PAC GP inducing inputs')
plt.xlabel('input $x$')
plt.ylabel('output $y$')
plt.grid()
plt.legend(loc=2)
plt.ylim([-1.5, 1.5])
plt.xlim([-3, 3])
