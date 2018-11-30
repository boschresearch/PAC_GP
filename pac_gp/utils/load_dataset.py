# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""

import sklearn.datasets


def load(dataset_name):
    if dataset_name == 'boston':
        data = sklearn.datasets.load_boston(return_X_y=True)
    else:
        raise Exception('Dataset %s is not known.' % dataset_name)

    return data
