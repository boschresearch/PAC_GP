# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""
import matplotlib
#matplotlib.use('agg')
import utils
import matplotlib.pylab as plt
import numpy as np
from utils import metrics
from copy import deepcopy as cp
from gp import kerns
import pandas as pd
import os
import itertools
import argparse
import pandas

import utils.plotting as plotting
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import utils.helpers as helpers
import utils.load_dataset as load_dataset


def run(dataset_name, fn_out, nInd_range, test_size=0.1, n_repetitions=3,
        ARD=False, epsilon=0.2, loss='01_loss'):
    """
    running methods

    input:
    dataset_name    :   name of dataset to load
    fn_out          :   name of results file
    nInd_range      :   list with number of inducing points
    test_size       :   fraction of test data points
    n_repetitions   :   number of repetitions
    ARD             :   automatic relevance determinant
    epsilon         :   sensitivity parameter of the loss function
    loss            :   loss function to use (01_loss, inv_gauss)
    """

    # load data
    X, y = load_dataset.load(dataset_name)
    Y = y[:, np.newaxis]

    # scale to zero mean and unit variance
    X = preprocessing.scale(X)
    Y = preprocessing.scale(Y)
    F = X.shape[1]

    data = []

    # vary number of inducing points
    for nInd in nInd_range:
        for i in range(n_repetitions):
            RV_vfe = helpers.compare(X, Y, 'GPflow VFE', seed=i,
                                     test_size=test_size, ARD=ARD, nInd=nInd,
                                     epsilon=epsilon, loss=loss)
            RV_fitc = helpers.compare(X, Y, 'GPflow FITC', seed=i,
                                      test_size=test_size, ARD=ARD, nInd=nInd,
                                      epsilon=epsilon, loss=loss)
            RV_pac = helpers.compare(X, Y, 'bkl-PAC Inducing Hyp GP', seed=i,
                                     test_size=test_size, ARD=ARD, nInd=nInd,
                                     epsilon=epsilon, loss=loss)
            RV_pac2 = helpers.compare(X, Y, 'sqrt-PAC Inducing Hyp GP', seed=i,
                                      test_size=test_size, ARD=ARD, nInd=nInd,
                                      epsilon=epsilon, loss=loss)

            data += RV_pac
            data += RV_vfe
            data += RV_fitc
            data += RV_pac2

    df = pd.DataFrame(data)
    df.to_pickle(fn_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running sparse GPs')

    parser.add_argument('-r', '--run', help='run', action='store_true',
                        default=False)
    parser.add_argument('-p', '--plot', help='plot', action='store_true',
                        default=False)
    parser.add_argument('-d', '--dataset', default='boston')
    parser.add_argument('-a', '--ARD', help='use ARD', action='store_true',
                        default=False)
    parser.add_argument('-t', '--test_size', help='set testsize in [0.0, 1.0]',
                        default=0.2)
    parser.add_argument('-e', '--epsilon', help='set epsilon', default=0.6)
    parser.add_argument('-n', '--n_reps', help='number of repetitions',
                        default=10)
    parser.add_argument('-s', '--size', help='number of inducing variables',
                        default='S')
    parser.add_argument('-l', '--loss', help='loss function',
                        default='01_loss')

    args = parser.parse_args()
    args.epsilon = float(args.epsilon)
    args.test_size = float(args.test_size)
    args.n_reps = int(args.n_reps)

    result_dir = 'results_rebuttal'
    fn_args = (args.dataset, args.loss, args.size, args.ARD, args.epsilon,
               100*args.test_size, args.n_reps)
    fn_base = '%s_%s_%s_ARD%d_eps%.1f_testsize%d_nReps%d' % fn_args
    fn_results = os.path.join('ind_points', '%s.pckl' % fn_base)
    if not(os.path.exists('ind_points')):
        os.mkdir('ind_points')

    if args.size == 'S':
        nInd_range = np.array([20, 40, 60, 80])
    elif args.size == 'M':
        nInd_range = np.array([50, 100, 150, 200, 250])
    elif args.size == 'L':
        nInd_range = np.array([100, 200, 300, 400, 500])

    if args.run:
        print('Run Experiments')
        run(args.dataset, fn_results, nInd_range,
            test_size=float(args.test_size), ARD=args.ARD,
            epsilon=args.epsilon, n_repetitions=args.n_reps, loss=args.loss)

    if args.plot:
        D = pandas.read_pickle(fn_results)
        models = ['bkl-PAC Inducing Hyp GP', 'sqrt-PAC Inducing Hyp GP',
                  'GPflow VFE', 'GPflow FITC']
        plotting.plot(D, models, x="nInd", xticks=[0.2, 0.4, 0.6, 0.8, 1.0],
                      ylim=(0, 0.85))
        fn_png = os.path.join('ind_points', '%s.png' % fn_base)
        fn_pdf = os.path.join('ind_points', '%s.pdf' % fn_base)
        plt.savefig(fn_png)
        plt.savefig(fn_pdf)
        plt.close()
