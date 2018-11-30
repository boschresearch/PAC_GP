# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch
"""
import gpflow


class GPflowWrapper():
    def __init__(self, model):
        self.model = model

    def optimize(self):
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(self.model)


class GPflowSparseWrapper(GPflowWrapper):
    pass


class GPflowFullWrapper(GPflowWrapper):
    pass
