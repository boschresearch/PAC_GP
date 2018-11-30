# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import matplotlib
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

labels = {'GPflow Full GP': 'full-GP',
          'bkl-PAC HYP GP': 'kl-PAC-GP',
          'sqrt-PAC HYP GP': 'sqrt-PAC-GP',
          'bkl-PAC Inducing Hyp GP': 'kl-PAC-SGP',
          'sqrt-PAC Inducing Hyp GP': 'sqrt-PAC-SGP',
          'GPflow VFE': 'VFE',
          'GPflow FITC': 'FITC'
          }

colors = {'bkl-PAC HYP GP': sns.color_palette("Paired")[0],
          'bkl-PAC Inducing Hyp GP': sns.color_palette("Paired")[0],
          'sqrt-PAC Inducing Hyp GP': sns.color_palette("Paired")[1],
          'sqrt-PAC HYP GP': sns.color_palette("Paired")[1],
          'GPflow Full GP': sns.color_palette("Paired")[4],
          'GPflow FITC': sns.color_palette("Paired")[3],
          'GPflow VFE': sns.color_palette("Paired")[2],
          }


def plot_lines(ax, D, x, metric, models, ylabel=None, xticks=None, ylim=None,
               yticks=None, legend=False):
    """
    beautfiy comparison plot
    """
    bars = []
    for i, model in enumerate(models):
        D_filtered = D[(D['model'] == model) & (D['metric'] == metric)]
        if metric == 'KL-divergence':
            D_filtered = D_filtered.reset_index()
            D_filtered['value'] = D_filtered['value'].div(D_filtered.N, axis=0)

        agg_params = ['size', 'mean', 'var']
        D_filtered = D_filtered.groupby(x).agg(agg_params).reset_index()
        mean = D_filtered['value']['mean']
        stderr = np.sqrt(D_filtered['value']['var'])
        stderr /= np.sqrt(D_filtered['value']['size'])

        _x = np.arange(len(mean))
        _bar = plt.bar(_x + 0.2*i, mean, label=labels[model], width=0.20,
                       yerr=stderr, color=colors[model],
                       ecolor='#696969')
        bars.append(_bar)

    plt.xticks(_x + 0.2, D_filtered[x])
    plt.xlabel(x)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylabel is not None:
        plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    if yticks is not None:
        plt.yticks(yticks)
    if legend:
        plt.legend(bbox_to_anchor=(-5.5, 1.02, 4, 0.1), loc=3, ncol=4,
                   mode="expand", borderaxespad=0., frameon=False)

    return bars


def plot(D, models, x, xticks=None, ylim=None, legend=True, yticks=None):
    """
    plotting results with respect to
    - upper-bound-bkl
    - gibbs-risk-train
    - gibbs-risk
    - MSE
    - KL-divergence
    """
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(left=0.05, bottom=0.2, wspace=0.40, right=1.0)

    ax = fig.add_subplot(151)
    plot_lines(ax, D, x, 'upper-bound-bkl', models, ylabel='Upper Bound',
               ylim=ylim, yticks=yticks)

    ax = fig.add_subplot(152)
    plot_lines(ax, D, x, 'gibbs-risk-train', models, ylabel='$R_S$[Train]',
               ylim=ylim, yticks=yticks)

    ax = fig.add_subplot(153)
    plot_lines(ax, D, x, 'gibbs-risk', models, ylabel='$R_S$[Test]',
               ylim=ylim, yticks=yticks)

    ax = fig.add_subplot(154)
    plot_lines(ax, D, x, 'MSE', models, ylabel='MSE')

    ax = fig.add_subplot(155)
    bars = plot_lines(ax, D, x, 'KL-divergence', models, ylabel='KL / N',
                      legend=legend)
