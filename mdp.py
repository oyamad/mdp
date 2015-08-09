"""
Filename: mdp.py

Author: Daisuke Oyama

Base class for solving Markov decision processes (MDP) with discrete
states and actions.

Markov Decision Processes
-------------------------

Solution Algorithms
-------------------

Error Bounds and Termination Conditions
---------------------------------------

References
----------
M. L. Puterman, Markov Decision Processes: Discrete Stochastic Dynamic
Programming, Wiley-Interscience, 2005.

"""
from __future__ import division
import numpy as np
import scipy.sparse as sp
from numba import jit
from quantecon import MarkovChain


class MDP(object):
    """
    Class for dealing with a Markov decision process (MDP) with n states
    and m actions.

    Work with state-action pairs. Sparse matrices
    supported. State-action pairs will be sorted in a lexicographic
    order. Below let L denote the number of feasible state-action pairs.

    Parameters
    ----------
    R : array_like(ndim=1)
        Array representing the reward function, of length L, where, if i
        corresponds to (s, a), R[i] is the flow reward when the current
        state is s and the action chosen is a.

    Q : array_like(ndim=2)
        Array representing the transition probabilities, of shape
        L x n, where, if i corresponds to (s, a), Q[i, s'] is the
        probability that the state in the next period is s' when the
        current state is s and the action chosen is a.

    s_indices : array_like(ndim=1)

    a_indices : array_like(ndim=1)

    beta : scalar(float)
        Discount factor. Must be in (0, 1).

    Attributes
    ----------
    R, Q, beta : see Parameters.

    num_sa_pairs : scalar(int)
        Number of state-action pairs.

    num_states : scalar(int)
        Number of states.

    num_actions : scalar(int)
        Number of actions.

    """
    def __init__(self, R, Q, beta, s_indices=None, a_indices=None):
        self._sa_pair = False
        self._sparse = False

        if sp.issparse(Q):
            self.Q = Q.tocsr()
            self._sa_pair = True
            self._sparse = True
        else:
            self.Q = np.asarray(Q)
            if self.Q.ndim == 2:
                self._sa_pair = True
            elif self.Q.ndim != 3:
                raise ValueError('Q must be 2- or 3-dimensional')

        self.R = np.asarray(R)
        if not (self.R.ndim in [1, 2]):
            raise ValueError('R must be 1- or 2-dimensional')

        msg_dimension = 'dimensions of R and Q must be either 1 and 2, ' + \
                        'of 2 and 3'
        msg_shape = 'shapes of R and Q must either (n, m) or (n, m, n), ' + \
                    'or (L,) and (L, n)'

        if self._sa_pair:
            self.num_sa_pairs, self.num_states = self.Q.shape

            if self.R.ndim != 1:
                raise ValueError(msg_dimension)
            if self.R.shape != (self.num_sa_pairs,):
                raise ValueError(msg_shape)

            if s_indices is None:
                raise ValueError('s_indices must be supplied')
            if a_indices is None:
                raise ValueError('a_indices must be supplied')
            if not (len(s_indices) == self.num_sa_pairs and
                    len(a_indices) == self.num_sa_pairs):
                raise ValueError(
                    'length of s_indices and a_indices must be equal to ' +
                    'the number of state-action pairs'
                )

            # Sort indices and elements of Q
            sa_ptrs = sp.coo_matrix(
                (np.arange(self.num_sa_pairs), (s_indices, a_indices))
            ).tocsr()
            sa_ptrs.sort_indices()
            self.a_indices = sa_ptrs.indices
            self.a_indptr = sa_ptrs.indptr
            self.num_actions = sa_ptrs.shape[1]
            if self._sparse:
                self.Q = sp.csr_matrix(self.Q[sa_ptrs.data])
            else:
                self.Q = self.Q[sa_ptrs.data]

            _s_indices = np.empty(self.num_sa_pairs,
                                  dtype=self.a_indices.dtype)
            for i in range(self.num_states):
                for j in range(self.a_indptr[i], self.a_indptr[i+1]):
                    _s_indices[j] = i
            self.s_indices = _s_indices

            self.R = self.R[sa_ptrs.data]

            # Define state-wise maximization
            def s_wise_max(vals, return_argmax=False):
                """
                Return the vector max_a vals(s, a), where vals is represented
                by a 1-dimensional ndarray of shape  (self.num_sa_pairs,).

                """
                out_max = np.empty(self.num_states)
                if return_argmax:
                    out_argmax = np.empty(self.num_states, dtype=int)
                    _s_wise_max_argmax(self.a_indices, self.a_indptr, vals,
                                       out_max=out_max, out_argmax=out_argmax)
                    return out_max, out_argmax
                else:
                    _s_wise_max(self.a_indices, self.a_indptr, vals,
                                out_max=out_max)
                    return out_max

            self.s_wise_max = s_wise_max

        else:  # Not self._sa_pair
            if self.R.ndim != 2:
                raise ValueError(msg_dimension)
            self.num_states, self.num_actions = self.R.shape

            if self.Q.shape != \
               (self.num_states, self.num_actions, self.num_states):
                raise ValueError(msg_shape)

            self.s_indices, self.a_indices = None, None
            self.num_sa_pairs = None

            # Define state-wise maximization
            def s_wise_max(vals, return_argmax=False):
                """
                Return the vector max_a vals(s, a), where vals is represented
                by a 2-dimensional ndarray of shape (self.num_states,
                self.num_actions).

                """
                if return_argmax:
                    out_argmax = vals.argmax(axis=1)
                    out_max = vals[np.arange(self.num_states), out_argmax]
                    return out_max, out_argmax
                else:
                    out_max = vals.max(axis=1)
                    return out_max

            self.s_wise_max = s_wise_max

        if not (0 < beta < 1):
            raise ValueError('beta must be in (0, 1)')
        self.beta = beta

        self.epsilon = 1e-3
        self.max_iter = 100
        self.tol = 1e-8

        # Linear equation solver to be used in evaluate_policy
        if self._sparse:
            self._lineq_solve = sp.linalg.spsolve
            self._I = sp.identity(self.num_states)
        else:
            self._lineq_solve = np.linalg.solve
            self._I = np.identity(self.num_states)

    def RQ_sigma(self, sigma):
        if self._sa_pair:
            sigma_indices = np.empty(self.num_states, dtype=int)
            _find_indices(self.a_indices, self.a_indptr, sigma,
                          out=sigma_indices)
            R_sigma, Q_sigma = self.R[sigma_indices], self.Q[sigma_indices]
        else:
            R_sigma = self.R[np.arange(self.num_states), sigma]
            Q_sigma = self.Q[np.arange(self.num_states), sigma]

        return R_sigma, Q_sigma

    def bellman_operator(self, w, compute_policy=False):
        """
        The Bellman operator, which computes and returns the updated
        value function Tw for a value function w.

        Parameters
        ----------
        w : array_like(float, ndim=1)
            Value function vector, of length n.

        compute_policy : bool, optional(default=False)
            Whether or not to additionally return the w-greedy policy.

        Returns
        -------
        Tw : array_like(float, ndim=1)
            Updated value function vector, of length n.

        sigma : array_like(int, ndim=1)
            w-greedy policy vector, of length n. Only returned if
            `compute_policy=True`.

        """
        vals = self.R + self.beta * self.Q.dot(w)  # Shape: (L,) or (n, m)

        if compute_policy:
            Tw, sigma = self.s_wise_max(vals, return_argmax=True)
            return Tw, sigma
        else:
            Tw = self.s_wise_max(vals)
            return Tw

    def T_sigma(self, sigma):
        """
        Return the T_sigma operator.

        Parameters
        ----------
        sigma : array_like(int, ndim=1)
            Policy vector, of length n.

        Returns
        -------
        callable
            The T_sigma operator.

        """
        R_sigma, Q_sigma = self.RQ_sigma(sigma)
        return lambda w: R_sigma + self.beta * Q_sigma.dot(w)

    def compute_greedy(self, w):
        """
        Compute the w-greedy policy.

        Parameters
        ----------
        w : array_like(float, ndim=1)
            Value function vector, of length n.

        Returns
        -------
        sigma : array_like(int, ndim=1)
            w-greedy policy vector, of length n.

        """
        _, sigma = self.bellman_operator(w, compute_policy=True)
        return sigma

    def evaluate_policy(self, sigma):
        """
        Compute the value of a policy.

        Parameters
        ----------
        sigma : array_like(int, ndim=1)
            Policy vector, of length n.

        Returns
        -------
        v_sigma : array_like(float, ndim=1)
            Value vector of `sigma`, of length n.

        """
        # Solve (I - beta * Q_sigma) v = R_sigma for v
        R_sigma, Q_sigma = self.RQ_sigma(sigma)
        b = R_sigma

        A = self._I - self.beta * Q_sigma

        v_sigma = self._lineq_solve(A, b)

        return v_sigma

    def solve(self, method='policy_iteration',
              w_0=None, epsilon=None, max_iter=None, return_num_iter=False,
              k=20):
        """
        Solve the dynamic programming problem.

        """
        if method in ['value_iteration', 'vi']:
            v_star, sigma_star, num_iter = \
                self.value_iteration(
                    w_0=w_0, epsilon=epsilon, max_iter=max_iter
                )
        elif method in ['policy_iteration', 'pi']:
            v_star, sigma_star, num_iter = \
                self.policy_iteration(w_0=w_0, max_iter=max_iter)
        elif method in ['modified_policy_iteration', 'mpi']:
            v_star, sigma_star, num_iter = \
                self.modified_policy_iteration(
                    w_0=w_0, epsilon=epsilon, max_iter=max_iter, k=k
                )
        else:
            raise ValueError('invalid method')

        mc = self.controlled_mc(sigma_star)

        if return_num_iter:
            return v_star, sigma_star, mc, num_iter
        else:
            return v_star, sigma_star, mc

    def successive_approx(self, T, w_0, tol, max_iter):
        # May be replaced with quantecon.compute_fixed_point
        if max_iter <= 0:
            return w_0, 0

        w = w_0
        for i in range(max_iter):
            new_w = T(w)
            if np.abs(new_w - w).max() < tol:
                w = new_w
                break
            w = new_w

        num_iter = i + 1

        return w, num_iter

    def value_iteration(self, w_0=None, epsilon=None, max_iter=None):
        if w_0 is None:
            w_0 = self.s_wise_max(self.R)
        if max_iter is None:
            max_iter = self.max_iter
        if epsilon is None:
            epsilon = self.epsilon

        tol = epsilon * (1-self.beta) / (2*self.beta)
        v, num_iter = \
            self.successive_approx(T=self.bellman_operator,
                                   w_0=w_0, tol=tol, max_iter=max_iter)
        sigma = self.compute_greedy(v)

        return v, sigma, num_iter

    def policy_iteration(self, w_0=None, max_iter=None):
        # What for initial condition?
        if w_0 is None:
            w_0 = self.s_wise_max(self.R)
        if max_iter is None:
            max_iter = self.max_iter

        sigma = self.compute_greedy(w_0)
        for i in range(max_iter):
            # Policy evaluation
            v_sigma = self.evaluate_policy(sigma)
            # Policy improvement
            new_sigma = self.compute_greedy(v_sigma)
            if np.array_equal(new_sigma, sigma):
                break
            sigma = new_sigma

        num_iter = i + 1

        return v_sigma, sigma, num_iter

    def modified_policy_iteration(self, w_0=None, epsilon=None, max_iter=None,
                                  k=20):
        if w_0 is None:
            w_0 = self.s_wise_max(self.R)
        if max_iter is None:
            max_iter = self.max_iter
        if epsilon is None:
            epsilon = self.epsilon

        def span(z):
            return z.max() - z.min()

        def midrange(z):
            return (z.min() + z.max()) / 2

        v = w_0
        for i in range(max_iter):
            # Policy improvement
            u, sigma = self.bellman_operator(v, compute_policy=True)
            diff = u - v
            if span(diff) < epsilon * (1-self.beta) / self.beta:
                v = u + midrange(diff) * self.beta / (1 - self.beta)
                break
            # Partial policy evaluation with k iterations
            v, _ = \
                self.successive_approx(T=self.T_sigma(sigma), w_0=u,
                                       tol=self.tol, max_iter=k-1)

        num_iter = i + 1

        return v, sigma, num_iter

    def controlled_mc(self, sigma):
        """
        Returns the controlled Markov chain for a given policy `sigma`.

        """
        _, Q_sigma = self.RQ_sigma(sigma)
        if self._sparse:
            Q_sigma = Q_sigma.toarray()
        return MarkovChain(Q_sigma)


@jit(nopython=True)
def _s_wise_max_argmax(a_indices, a_indptr, vals, out_max, out_argmax):
    n = len(out_max)
    for i in range(n):
        if a_indptr[i] != a_indptr[i+1]:
            m = a_indptr[i]
            for j in range(a_indptr[i]+1, a_indptr[i+1]):
                if vals[j] > vals[m]:
                    m = j
            out_max[i] = vals[m]
            out_argmax[i] = a_indices[m]


@jit(nopython=True)
def _s_wise_max(a_indices, a_indptr, vals, out_max):
    n = len(out_max)
    for i in range(n):
        if a_indptr[i] != a_indptr[i+1]:
            m = a_indptr[i]
            for j in range(a_indptr[i]+1, a_indptr[i+1]):
                if vals[j] > vals[m]:
                    m = j
            out_max[i] = vals[m]


@jit(nopython=True)
def _find_indices(a_indices, a_indptr, sigma, out):
    n = len(sigma)
    for i in range(n):
        for j in range(a_indptr[i], a_indptr[i+1]):
            if sigma[i] == a_indices[j]:
                out[i] = j
