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

from scipy.optimize import minimize

from tensorflow.contrib.distributions import MultivariateNormalFullCovariance as MVN
from tensorflow.contrib.distributions import kl_divergence as KL

from pac_gp.gp.gpr import GPRFITC
from pac_gp.gp.gpr import GPR
from pac_gp.gp.kerns import RBF
from pac_gp.gp.mean_functions import Zero

from pac_gp.utils.bin_kl import BinaryKLInv
from pac_gp.utils.utils import expand_vector
from pac_gp.utils.utils import flatten
from pac_gp.utils.utils import Configurable
from pac_gp.utils.utils import variable_summaries
from pac_gp.utils.utils import clamp_and_round
from pac_gp.utils.transformations import Log1pe


class PAC_GP_BASE(Configurable):

    def __init__(self, X, Y, sn2, kernel=None, mean_function=None,
                 epsilon=0.2, delta=0.01, verbosity=0, method='bkl',
                 loss='01_loss'):

        assert X.ndim == 2
        assert Y.ndim == 2
        assert X.shape[0] == Y.shape[0]

        self.X = np.array(X)
        self.Y = np.array(Y)

        _, self.input_dim = self.X.shape
        _, self.output_dim = self.Y.shape

        self.kernel = kernel or RBF(self.input_dim)
        self.mean_function = mean_function or Zero()

        self.epsilon = epsilon
        self.delta = delta

        self.sn2 = sn2

        self.jitter = np.asarray(1e-6, dtype=np.float64)

        self.method = method  # can be 'bkl' or 'naive'
        self.loss = loss

        # Optimization options
        self.maxiter = 1000
        self.maxfun = 1000
        self.disp = False
        self.ftol = 1e-8
        self.gtol = 1e-8

        # Hyper parameter rounding
        self.rounding_digits = int(2)
        self.min_log = -6.0
        self.max_log = 6.0
        self.num_hyps = int(len(self.kernel.lengthscale) + 1)
        self.round_hyps_flag = True

        self.debug = False
        self.verbosity = verbosity
        self.ninits = 0

        self.pos_trans = Log1pe()

        if self.verbosity > 3:
            self.debug = True
            self.disp = True
        if self.verbosity > 0:
            self.logdir = '_tf_logs_gp'
            # Tensorboard file writer for training and test (e.g. train_path = 'train/')
            self.summary_calls = 0
        self.debug_variables = 0
        self._generate_ops()

    def predict(self, Xnew, full_cov=False):
        """ GP prediction at given input points

        p(y* | x*, X, y) \sim \mathcal{N}(y* | mean, var)

        Args:
            Xnew: ndarray (num_points, input_dim), input points matrix
            full_cov: bool, flat whether to return full covariance or only
                      diagonal

        Returns:
            mean: ndarray (num_points, output_dim), mean of GP prediction
            var: ndarray (num_points, num_points, output_dim) for full
                 covariance (full_cov=True) or (num_points, output_dim)
                 for diagonal (full_cov=False)
        """
        with tf.Session() as sess:
            feed = self._get_prediction_feed(Xnew)
            if full_cov is True:
                mean, var = sess.run([self.pred_mean_y, self.pred_cov_y],
                                     feed_dict=feed)
            else:
                mean, var = sess.run([self.pred_mean_y, self.pred_var_y],
                                     feed_dict=feed)
        return mean, var

    def predict_noiseless(self, Xnew, full_cov=False):
        """ GP prediction of the latent function at given input points

        p(f* | x*, X, y) \sim \mathcal{N}(f* | mean, var)

        Args:
            Xnew: ndarray (num_points, input_dim), input points matrix
            full_cov: bool, flat whether to return full covariance or only
                      diagonal

        Returns:
            mean: ndarray (num_points, output_dim), mean of GP prediction
            var: ndarray (num_points, num_points, output_dim) for full
                 covariance (full_cov=True) or (num_points, output_dim)
                 for diagonal (full_cov=False)
        """
        with tf.Session() as sess:
            feed = self._get_prediction_feed(Xnew)
            if full_cov is True:
                mean, var = sess.run([self.pred_mean_f, self.pred_cov_f],
                                     feed_dict=feed)
            else:
                mean, var = sess.run([self.pred_mean_f, self.pred_var_f],
                                     feed_dict=feed)
        return mean, var

    def get_PAC_info(self):
        variables = [self.upper_bound,
                     self.upper_bound_bkl,
                     self.empirical_risk,
                     self.KL_divergence,
                     self.penalty,
                     self.penalty_bkl]
        with tf.Session() as sess:
            feed = self._get_PAC_feed()
            return sess.run(variables, feed_dict=feed)

    def get_empirical_risk(self):
        with tf.Session() as sess:
            feed = self._get_PAC_feed()
            return sess.run(self.empirical_risk, feed_dict=feed)

    def get_upper_bound(self):
        with tf.Session() as sess:
            feed = self._get_PAC_feed()
            return sess.run(self.upper_bound, feed_dict=feed)

    def get_upper_bound_bkl(self):
        with tf.Session() as sess:
            feed = self._get_PAC_feed()
            return sess.run(self.upper_bound_bkl, feed_dict=feed)

    def get_KL_divergence(self):
        with tf.Session() as sess:
            feed = self._get_PAC_feed()
            return sess.run(self.KL_divergence, feed_dict=feed)

    def get_penalty(self):
        with tf.Session() as sess:
            feed = self._get_PAC_feed()
            return sess.run(self.penalty, feed_dict=feed)

    def get_log_theta(self):
        with tf.Session() as sess:
            feed = self._get_PAC_feed()
            return sess.run(self.log_Theta, feed_dict=feed)

    def get_penalty_bkl(self):
        with tf.Session() as sess:
            feed = self._get_PAC_feed()
            return sess.run(self.penalty_bkl, feed_dict=feed)

    def optimize(self):
        x0 = self._get_optimization_x0()
        bounds = self._get_bounds()

        if self.verbosity > 0:
            self.ninits += 1
            self.writer_tf = tf.summary.FileWriter(self.logdir + '_run_%d' % (self.ninits))
            self.summary_calls = 0

        with tf.Session() as sess:
            def objective_fcn(x):
                feed = self._get_optimization_feed(x)
                J = sess.run([self.objective], feed_dict=feed)
                if (self.verbosity > 0) and (self.debug_variables > 0):
                    stats = sess.run([self._debug_op], feed_dict=feed)[0]
                    self.writer_tf.add_summary(stats, self.summary_calls+1)
                    self.summary_calls += 1
                return J

            def objective_grad_fcn(x):
                feed = self._get_optimization_feed(x)
                return flatten(sess.run(self.objective_grad, feed_dict=feed))

            def debug_callback(x):
                if self.debug is True:
                    debug_vars = [self.upper_bound, self.upper_bound_bkl,
                                  self.empirical_risk, self.KL_divergence,
                                  self.penalty, self.penalty_bkl]
                    debug_names = ['upper_bound', 'upper_bound_bkl',
                                   'empirical_risk', 'KL_divergence',
                                   'penalty', 'penalty_bkl']
                    feed = self._get_optimization_feed(x)
                    res = sess.run(debug_vars, feed_dict=feed)
                    print(['%s: %f ' % (name, value) for name, value in zip(debug_names, res)])

            opts = {
                    'maxiter': self.maxiter,
                    'maxfun': self.maxfun,
                    'disp': self.disp,
                    'ftol': self.ftol,
                    'gtol': self.gtol
                   }

            # Optimize objective using BFGS
            res = minimize(fun=objective_fcn,
                           x0=x0,
                           method='L-BFGS-B',
                           jac=objective_grad_fcn,
                           hess=None,
                           callback=debug_callback,
                           options=opts,
                           bounds = bounds)
        self._set_variables(res.x)

    def configure(self, config):
        """ Configure class instance based on keyword/values in config dict

        The computation graph is rebuild once a new configure is called.

        Args:
            config: dict, values to be written in member attributes of given key/name
        """
        super(PAC_GP_BASE, self).configure(config)

    def _generate_ops(self):
        # Set up general TF graph for PAC-GP optimization
        self._generate_data_ops()
        # This is a GP specific abstract method.
        # To be implemented by each subclass
        self._generate_gp_ops()

        self._generate_prediction_ops()
        # Generate PAC bound ops
        self._generate_pac_ops()
        # Objective and objective gradient ops
        # To be implemented by each subclass
        self._generate_optimization_ops()

        # Common op to run all variable summaries

    def _generate_data_ops(self):
        """ Generate computation graph for sparse PAC-GP
        """
        # PAC parameters
        self.epsilon_tf = tf.placeholder(dtype=tf.float64,
                                         shape=(), name='epsilon')
        self.delta_tf = tf.placeholder(dtype=tf.float64,
                                       shape=(), name='delta')

        # GP training data: X (N, D), Y (N, 1)
        self.X_tf = tf.placeholder(dtype=tf.float64,
                                   shape=(None, self.input_dim),
                                   name='X')
        self.Y_tf = tf.placeholder(dtype=tf.float64,
                                   shape=(None, self.output_dim),
                                   name='Y')

        # GP test inputs: X_new (N_new, D)
        self.Xnew_tf = tf.placeholder(dtype=tf.float64,
                                      shape=(None, self.input_dim),
                                      name='Xnew')

        # GP hyperparameter
        self.sn2_unc_tf = tf.placeholder(dtype=tf.float64,
                                         shape=(1, ),
                                         name='sn2_unconstrained')
        self.sn2_tf = self.pos_trans.forward_tensor(self.sn2_unc_tf)

        # Hyper parameter rounding limits
        self.rounding_digits_tf = tf.placeholder(dtype=tf.int32, shape=(), name='round_digits')
        self.min_log_tf = tf.placeholder(dtype=tf.float64, shape=(), name='min_log')
        self.max_log_tf = tf.placeholder(dtype=tf.float64, shape=(), name='max_log')
        self.num_hyps_tf = tf.placeholder(dtype=tf.int32, shape=(), name='num_hyps')

        # Add summary ops for all relevant variables for tensorboard visualization
        if self.verbosity >0:
            #variable_summaries(tf.squeeze(self.sn2_unc_tf), name='noise_variance', vector=False)
            self.debug_variables=0
            self._debug_op = tf.summary.merge_all(key="stats_summaries")

        # variable_summaries(self.kernel.variance, 'kernel variance', vector=False)

    def _generate_prediction_ops(self):
        """ GP predictive distributions for new inputs Xnew_tf
            f = noisefree
            y = noisy

            cov = full covariance matrix
            var = only diagonal entries
        """
        self.pred_mean_f, self.pred_cov_f = self.gp._build_predict_f(self.Xnew_tf, full_cov=True)
        self.pred_mean_y, self.pred_cov_y = self.gp._build_predict_y(self.Xnew_tf, full_cov=True)

        _, self.pred_var_f = self.gp._build_predict_f(self.Xnew_tf,
                                                      full_cov=False)
        _, self.pred_var_y = self.gp._build_predict_y(self.Xnew_tf,
                                                      full_cov=False)

        # GP predictive distribution for training inputs
        self.train_mean, self.train_var = self.gp._build_predict_f(self.X_tf, full_cov=False)
        # Remove singleton dimensions
        self.train_mean = tf.squeeze(self.train_mean)           # ONLY VALID FOR output_dim == 1
        self.train_var = tf.squeeze(self.train_var)             # ONLY VALID FOR output_dim == 1

    def _generate_pac_ops(self):
        # Multivariate normal distribution (GP prior/posterior)
        self.P_dist = MVN(loc=self.P_mean,
                          covariance_matrix=self.P_cov,
                          name='prior')
        self.Q_dist = MVN(loc=self.Q_mean,
                          covariance_matrix=self.Q_cov,
                          name='posterior')

        # KL(Q_dist||P_dist)
        self.KL_divergence = KL(self.Q_dist, self.P_dist)

        # Empirical risk based on GP prediction on training data set
        if self.loss=='01_loss':
            self.predictive_dist = tf.distributions.Normal(loc=self.train_mean, scale=tf.sqrt(self.train_var))
            self.cdf_y_p_eps = self.predictive_dist.cdf(tf.squeeze(self.Y_tf) + self.epsilon_tf)
            self.cdf_y_m_eps = self.predictive_dist.cdf(tf.squeeze(self.Y_tf) - self.epsilon_tf)
            self.empirical_risk = 1.0 - tf.reduce_mean(self.cdf_y_p_eps - self.cdf_y_m_eps)
        elif self.loss=='inv_gauss':
            #import pdb; pdb.set_trace()
            sf = tf.exp(-(self.train_mean-tf.squeeze(self.Y_tf))**2/(2.*self.train_var + self.epsilon_tf**2))
            inv_gauss = 1./tf.sqrt(1 + 2*self.train_var/self.epsilon_tf**2) * sf
            self.empirical_risk = 1 - tf.reduce_mean(inv_gauss)

        # Regularization term
        N = tf.cast(tf.shape(self.X_tf)[0], tf.float64)

        self.log_Theta = tf.cast(self.num_hyps_tf,dtype=tf.float64) * \
                         tf.log(1. + (self.max_log_tf - self.min_log_tf) *
                                (10. ** tf.cast(self.rounding_digits_tf, dtype=tf.float64)))

        self.penalty_1 = self.KL_divergence
        self.penalty_2 = self.log_Theta
        self.penalty_3 = tf.log((2 * tf.sqrt(N)) / self.delta_tf)

        self.penalty = tf.sqrt((self.penalty_1 + self.penalty_2 + self.penalty_3) / (2*N))
        self.penalty_bkl = (self.penalty_1 + self.penalty_2 + self.penalty_3) / N

        # Resulting risk upper bound
        self.upper_bound = self.empirical_risk + self.penalty
        self.upper_bound_bkl = BinaryKLInv(self.empirical_risk, self.penalty_bkl)

    def _get_PAC_feed(self):
        feed = {
                self.X_tf: self.X,
                self.Y_tf: self.Y,
                self.epsilon_tf: self.epsilon,
                self.delta_tf: self.delta,
                self.kernel.variance_unc_tf: self.pos_trans.backward(self.kernel.variance),
                self.kernel.lengthscales_unc_tf: self.pos_trans.backward(self.kernel.lengthscale),
                self.sn2_unc_tf: self.pos_trans.backward(self.sn2),
                self.num_hyps_tf: self.num_hyps,
                self.min_log_tf: self.min_log,
                self.max_log_tf: self.max_log,
                self.rounding_digits_tf: self.rounding_digits
               }
        return feed

    def _get_optimization_feed(self):
        pass

    def _get_prediction_feed(self, Xnew):
        feed = {
                self.Xnew_tf: Xnew,
                self.X_tf: self.X,
                self.Y_tf: self.Y,
                self.epsilon_tf: self.epsilon,
                self.delta_tf: self.delta,
                self.kernel.variance_unc_tf: self.pos_trans.backward(self.kernel.variance),
                self.kernel.lengthscales_unc_tf: self.pos_trans.backward(self.kernel.lengthscale),
                self.sn2_unc_tf: self.pos_trans.backward(self.sn2),
                self.num_hyps_tf: self.num_hyps,
                self.min_log_tf: self.min_log,
                self.max_log_tf: self.max_log,
                self.rounding_digits_tf: self.rounding_digits
               }
        return feed

    def round_hyps(self):
        if self.round_hyps_flag is True:
            self.kernel.lengthscale = clamp_and_round(self.kernel.lengthscale,
                                                      self.min_log, self.max_log,
                                                      self.rounding_digits)
            self.kernel.variance = clamp_and_round(self.kernel.variance,
                                                   self.min_log, self.max_log,
                                                   self.rounding_digits)


class PAC_FULL_GP_BASE(PAC_GP_BASE):

    def __init__(self, *args, **kwargs):
        super(PAC_FULL_GP_BASE, self).__init__(*args, **kwargs)

    def _generate_gp_ops(self):
        # Set up GP model
        self.gp = GPR(self.X_tf, self.Y_tf, self.sn2_tf,
                      self.kernel, self.mean_function)

        # P: GP prior (based on training data X)
        self.P_mean = self.mean_function(self.X_tf)    # (num_data, output_dim)
        self.P_mean = tf.squeeze(self.P_mean)          # ONLY VALID FOR output_dim == 1
        self.P_cov = self.kernel.K(self.X_tf)          # (num_data, num_data)
        self.P_cov += tf.eye(tf.shape(self.X_tf)[0], dtype=tf.float64) * self.jitter

        # Q: GP posterior
        self.Q_mean, self.Q_cov = self.gp._build_predict_f(self.X_tf, full_cov=True)
        self.Q_mean = tf.squeeze(self.Q_mean)           # ONLY VALID FOR output_dim == 1
        self.Q_cov = tf.squeeze(self.Q_cov)             # ONLY VALID FOR output_dim == 1
        self.Q_cov += tf.eye(tf.shape(self.X_tf)[0], dtype=tf.float64) * self.jitter


class PAC_SPARSE_GP_BASE(PAC_GP_BASE):

    def __init__(self, X, Y, Z, sn2, kernel=None, mean_function=None,
                 epsilon=0.2, delta=0.01, verbosity=0, method='bkl',loss='01_loss'):
        assert Z.ndim == 2
        self.Z = Z
        self.num_inducing, _ = self.Z.shape

        super(PAC_SPARSE_GP_BASE, self).__init__(X, Y, sn2, kernel,
                                                 mean_function, epsilon, delta, method=method, loss=loss)
        self.verbosity = verbosity

    def _generate_gp_ops(self):
        # GP inducing inputs
        self.Z_tf = tf.placeholder(dtype=tf.float64,
                                   shape=(self.num_inducing, self.input_dim),
                                   name='Z')

        # Set up GP model
        self.gp = GPRFITC(self.X_tf, self.Y_tf, self.sn2_tf,
                          self.kernel, self.mean_function, self.Z_tf)

        # P: GP prior (based on training data X)
        self.P_mean = self.mean_function(self.Z_tf)    # (num_data, output_dim)
        self.P_mean = tf.squeeze(self.P_mean)          # ONLY VALID FOR output_dim == 1
        self.P_cov = self.kernel.K(self.Z_tf)          # (num_data, num_data)
        self.P_cov += tf.eye(self.num_inducing, dtype=tf.float64) * self.jitter

        # Q: GP posterior
        self.Q_mean, self.Q_cov = self.gp._build_predict_f(self.Z_tf, full_cov=True)
        self.Q_mean = tf.squeeze(self.Q_mean)           # ONLY VALID FOR output_dim == 1
        self.Q_cov = tf.squeeze(self.Q_cov)             # ONLY VALID FOR output_dim == 1
        self.Q_cov += tf.eye(self.num_inducing, dtype=tf.float64) * self.jitter


    def _get_PAC_feed(self):
        feed = super(PAC_SPARSE_GP_BASE, self)._get_PAC_feed()
        feed[self.Z_tf] = self.Z
        return feed

    def _get_optimization_feed(self):
        pass

    def _get_prediction_feed(self, Xnew):
        feed = super(PAC_SPARSE_GP_BASE, self)._get_prediction_feed(Xnew)
        feed[self.Z_tf] = self.Z
        return feed

# Actual PAC GP implementations based on the base classes above
# These classes should be instantiated


class PAC_HYP_GP(PAC_FULL_GP_BASE):
    """ Full GP optimizing kernel hyperparameters
    """
    def __init__(self, *args, **kwargs):
        super(PAC_HYP_GP, self).__init__(*args, **kwargs)

    def _get_parameter_shapes(self):
        # Kernel lengthscale shape
        if self.kernel.ARD is True:
            lengthscales_shape = (self.input_dim, )
        else:
            lengthscales_shape = (1, )

        shapes = [lengthscales_shape,
                  (1, ),
                  (1, )]
        return shapes

    def _get_optimization_x0(self):
        # Reshape kernel hyperparameters (input_dim + 2)
        # into 1D optimization vector x
        variables = [self.pos_trans.backward(self.kernel.lengthscale),
                     self.pos_trans.backward(self.kernel.variance),
                     self.pos_trans.backward(self.sn2)]
        variables = flatten(variables)
        return variables

    def _get_bounds(self):
        x_min = self.pos_trans.backward(np.exp(self.min_log))
        x_max = self.pos_trans.backward(np.exp(self.max_log))

        if self.kernel.lengthscale.ndim==0:
            bounds = 2 * [[x_min, x_max]]
        else:
            bounds = (self.kernel.lengthscale.shape[0]+1) * [[x_min, x_max]]
        bounds.append([None, None])
        return bounds

    def _get_optimization_feed(self, x):
        # Distribute 1D optimization vector x
        # into individual model variables
        shapes = self._get_parameter_shapes()
        res = expand_vector(x, shapes)

        feed = {
                self.X_tf: self.X,
                self.Y_tf: self.Y,
                self.epsilon_tf: self.epsilon,
                self.delta_tf: self.delta,
                self.kernel.variance_unc_tf: res[1],
                self.kernel.lengthscales_unc_tf: res[0],
                self.sn2_unc_tf: res[2],
                self.num_hyps_tf: self.num_hyps,
                self.min_log_tf: self.min_log,
                self.max_log_tf: self.max_log,
                self.rounding_digits_tf: self.rounding_digits
               }
        return feed

    def _set_variables(self, x):
        shapes = self._get_parameter_shapes()
        res = expand_vector(x, shapes)

        self.kernel.lengthscale = self.pos_trans.forward(res[0])
        self.kernel.variance = self.pos_trans.forward(res[1])
        self.sn2 = self.pos_trans.forward(res[2])
        self.round_hyps()

    def _generate_optimization_ops(self):
        # I'M PRETTY SURE WE NEED THE GRADIENT WITH RESPECT TO UNCONSTRAINED PARAMS
        # variables = [self.kernel.lengthscales_tf,
        #              self.kernel.variance_tf,
        #              self.sn2_tf]
        variables = [self.kernel.lengthscales_unc_tf,
                     self.kernel.variance_unc_tf,
                     self.sn2_unc_tf]
        if self.method == 'bkl':
            self.objective = self.upper_bound_bkl
        else:
            self.objective = self.upper_bound
        self.objective_grad = tf.gradients(self.objective, variables)


class PAC_INDUCING_GP(PAC_SPARSE_GP_BASE):
    """ FITC sparse GP optimizing inducing inputs
    """

    def __init__(self, *args, **kwargs):
        super(PAC_INDUCING_GP, self).__init__(*args, **kwargs)
        self.num_hyps = 0

    def _get_optimization_x0(self):
        # Reshape inducing inputs Z (num_inducing, input_dim)
        # into 1D optimization vector x
        variables = [self.Z]
        return flatten(variables)

    def _get_optimization_feed(self, x):
        # Distribute 1D optimization vector x
        # into individual model variables
        shapes = [(self.num_inducing, self.input_dim)]
        res = expand_vector(x, shapes)

        feed = {
                self.X_tf: self.X,
                self.Y_tf: self.Y,
                self.epsilon_tf: self.epsilon,
                self.delta_tf: self.delta,
                self.kernel.variance_unc_tf: self.pos_trans.backward(self.kernel.variance),
                self.kernel.lengthscales_unc_tf: self.pos_trans.backward(self.kernel.lengthscale),
                self.sn2_unc_tf: self.pos_trans.backward(self.sn2),
                self.Z_tf: res[0],
                self.num_hyps_tf: self.num_hyps,
                self.min_log_tf: self.min_log,
                self.max_log_tf: self.max_log,
                self.rounding_digits_tf: self.rounding_digits
               }
        return feed

    def _set_variables(self, x):
        shapes = [(self.num_inducing, self.input_dim)]
        res = expand_vector(x, shapes)
        self.Z = res[0]

    def _generate_optimization_ops(self):
        variables = [self.Z_tf]
        if self.method == 'bkl':
            self.objective = self.upper_bound_bkl
        else:
            self.objective = self.upper_bound
        self.objective_grad = tf.gradients(self.objective, variables)


class PAC_INDUCING_HYP_GP(PAC_SPARSE_GP_BASE):
    """ FITC sparse GP optimizing inducing inputs
    and kernel hyperparameters
    """

    def __init__(self, *args, **kwargs):
        super(PAC_INDUCING_HYP_GP, self).__init__(*args, **kwargs)

    def _get_parameter_shapes(self):
        if self.kernel.ARD is True:
            shapes = [(self.num_inducing, self.input_dim),
                      (self.input_dim, ),
                      (1, ),
                      (1, )]
        else:
            shapes = [(self.num_inducing, self.input_dim),
                      (1, ),
                      (1, ),
                      (1, )]
        return shapes

    def _get_bounds(self):
        x_min = self.pos_trans.backward(np.exp(self.min_log))
        x_max = self.pos_trans.backward(np.exp(self.max_log))

        shapes = self._get_parameter_shapes()
        # adding bounds only for kernel parameters, not for inducing points or sn2
        bounds = np.sum([[[None, None]] * np.prod(s) if ((i==0) or (i==len(shapes)-1))
                         else [[x_min, x_max]] * np.prod(s) for i,s in enumerate(shapes)])
        return bounds

    def _get_optimization_x0(self):
        # Reshape inducing inputs Z (num_inducing, input_dim)
        # and kernel hyperparameters (input_dim + 2)
        # into 1D optimization vector x
        variables = [self.Z,
                     self.pos_trans.backward(self.kernel.lengthscale),
                     self.pos_trans.backward(self.kernel.variance),
                     self.pos_trans.backward(self.sn2)]
        return flatten(variables)

    def _get_optimization_feed(self, x):
        # Distribute 1D optimization vector x
        # into individual model variables
        shapes = self._get_parameter_shapes()
        res = expand_vector(x, shapes)

        feed = {
                self.X_tf: self.X,
                self.Y_tf: self.Y,
                self.epsilon_tf: self.epsilon,
                self.delta_tf: self.delta,
                self.kernel.variance_unc_tf: res[2],
                self.kernel.lengthscales_unc_tf: res[1],
                self.sn2_unc_tf: res[3],
                self.Z_tf: res[0],
                self.num_hyps_tf: self.num_hyps,
                self.min_log_tf: self.min_log,
                self.max_log_tf: self.max_log,
                self.rounding_digits_tf: self.rounding_digits
               }
        return feed

    def _set_variables(self, x):
        shapes = self._get_parameter_shapes()
        res = expand_vector(x, shapes)

        self.Z = res[0]
        self.kernel.lengthscale = self.pos_trans.forward(res[1])
        self.kernel.variance = self.pos_trans.forward(res[2])
        self.sn2 = self.pos_trans.forward(res[3])
        self.round_hyps()

    def _generate_optimization_ops(self):
        # variables = [self.Z_tf,
        #              self.kernel.lengthscales_tf,
        #              self.kernel.variance_tf,
        #              self.sn2_tf]
        variables = [self.Z_tf,
                     self.kernel.lengthscales_unc_tf,
                     self.kernel.variance_unc_tf,
                     self.sn2_unc_tf]

        if self.method == 'bkl':
            self.objective = self.upper_bound_bkl
        else:
            self.objective = self.upper_bound
        self.objective_grad = tf.gradients(self.objective, variables)
        if self.verbosity > 1:

            # Add summary ops for all relevant variables for tensorboard visualization
            variable_summaries(self.Z_tf, 'sparse_gp_inducing_inputs', vector=True)
            self.debug_variables += 1
            # Common op to run all variable summaries
            self._debug_op = tf.summary.merge_all(key="stats_summaries")
