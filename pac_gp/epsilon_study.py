"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""
import sys
import matplotlib
#matplotlib.use('agg')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import os
import argparse
from sklearn import preprocessing

import utils.plotting as plotting
import utils.load_dataset as load_dataset
import utils.helpers as helpers


def run(dataset_name, fn_out, epsilon_range, test_size=0.1, n_repetitions=10,
        ARD=False, nInd=0, loss='01_loss'):
    """
    running methods

    input:
    dataset_name    :   name of dataset to load
    fn_out          :   name of results file
    epsilon_range   :   list of epsilons
    test_size       :   fraction of test data points
    n_repetitions   :   number of repetitions
    ARD             :   automatic relevance determinant
    epsilon         :   sensitivity parameter of the loss function
    loss            :   loss function to use (01_loss, inv_gauss)
    """

    # load data
    X, y = load_dataset.load(dataset_name)
    Y = y[:, np.newaxis]

    print(dataset_name)
    print(X.shape)
    print(Y.shape)

    # scale to zero mean and unit variance
    X = preprocessing.scale(X)
    Y = preprocessing.scale(Y)
    F = X.shape[1]

    data = []
    for i in range(n_repetitions):
        # vary epsilon
        for ie, epsilon in enumerate(epsilon_range):

            if nInd == 0:
                # exact GP
                RV_gpflow = helpers.compare(X, Y, 'GPflow Full GP', seed=i,
                                            test_size=test_size, ARD=ARD,
                                            epsilon=epsilon, loss=loss)
                RV_pac = helpers.compare(X, Y, 'bkl-PAC HYP GP', seed=i,
                                         test_size=test_size, ARD=ARD,
                                         epsilon=epsilon, loss=loss)
                RV_naive = helpers.compare(X, Y, 'sqrt-PAC HYP GP', seed=i,
                                           test_size=test_size, ARD=ARD,
                                           epsilon=epsilon, loss=loss)
                RVs = [RV_pac, RV_naive, RV_gpflow]

            else:
                # sparse GP
                RV_vfe = helpers.compare(X, Y, 'GPflow VFE', seed=i,
                                         test_size=test_size, ARD=ARD,
                                         nInd=nInd, epsilon=epsilon, loss=loss)
                RV_fitc = helpers.compare(X, Y, 'GPflow FITC', seed=i,
                                          test_size=test_size, ARD=ARD,
                                          nInd=nInd, epsilon=epsilon,
                                          loss=loss)
                RV_pac = helpers.compare(X, Y, 'bkl-PAC Inducing Hyp GP',
                                         seed=i, test_size=test_size, ARD=ARD,
                                         nInd=nInd, epsilon=epsilon, loss=loss)
                RV_pac2 = helpers.compare(X, Y, 'sqrt-PAC Inducing Hyp GP',
                                          seed=i, test_size=test_size, ARD=ARD,
                                          nInd=nInd, epsilon=epsilon,
                                          loss=loss)
                RVs = [RV_vfe, RV_fitc, RV_pac, RV_pac2]

            for RV in RVs:
                data += RV

    df = pd.DataFrame(data)
    df.to_pickle(fn_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running full GPs')
    parser.add_argument('-r', '--run', help='run', action='store_true',
                        default=False)
    parser.add_argument('-p', '--plot', help='plot', action='store_true',
                        default=False)
    parser.add_argument('-d', '--dataset', default='boston')
    parser.add_argument('-a', '--ARD', help='use ARD', action='store_true',
                        default=False)
    parser.add_argument('-t', '--test_size', help='testsize in [0.0, 1.0]',
                        default=0.2)
    parser.add_argument('-n', '--n_reps', help='number of repetitions',
                        default=10)
    parser.add_argument('-m', '--nInd', help='number of inducing points',
                        default=0)
    parser.add_argument('-l', '--loss', help='loss function',
                        default='01_loss')

    args = parser.parse_args()
    args.test_size = float(args.test_size)
    args.n_reps = int(args.n_reps)
    args.nInd = int(args.nInd)

    dir_results = 'epsilon'
    if args.nInd == 0:
        models = ['bkl-PAC HYP GP', 'sqrt-PAC HYP GP', 'GPflow Full GP']
        fn_args = (args.dataset, args.loss, args.ARD, 100.*args.test_size,
                   args.n_reps)
        fn_base = '%s_%s_ARD%d_testsize%d_nReps%d' % fn_args

        fn_results = os.path.join(dir_results, '%s.pckl' % fn_base)
        fn_png = os.path.join(dir_results, '%s.png' % fn_base)
        fn_pdf = os.path.join(dir_results, '%s.pdf' % fn_base)
    else:
        models = ['bkl-PAC Inducing Hyp GP', 'sqrt-PAC Inducing Hyp GP',
                  'GPflow VFE', 'GPflow FITC']
        fn_args = (args.dataset, args.loss, args.nInd, args.ARD,
                   100.*args.test_size, args.n_reps)
        fn_base = '%s_%s_nInd%d_ARD%d_testsize%d_nReps%d' % fn_args
        fn_results = os.path.join(dir_results, '%s.pckl' % fn_base)
        fn_png = os.path.join(dir_results, '%s.png' % fn_base)
        fn_pdf = os.path.join(dir_results, '%s.pdf' % fn_base)

    if not(os.path.exists(dir_results)):
        os.mkdir(dir_results)

    epsilon_range = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    if args.run:
        print('Run Experiments')
        run(args.dataset, fn_results, epsilon_range,
            test_size=float(args.test_size), ARD=args.ARD,
            n_repetitions=args.n_reps, nInd=args.nInd, loss=args.loss)

    if args.plot:
        matplotlib.rc('font', **{'size': 14})
        D = pd.read_pickle(fn_results)
        plotting.plot(D, models, x="epsilon", xticks=[0.2, 0.4, 0.6, 0.8, 1.0],
                      ylim=(0, 0.85))
        plt.savefig(fn_png)
        plt.savefig(fn_pdf)
        plt.close()
