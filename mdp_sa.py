"""
Filename: mdp_sa.py

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
from random_stochmatrix import gen_random_stochmatrix, \
                               _random_probvec, _random_indices


class MDP_sa(object):
    """
    Class for dealing with a Markov decision process (MDP) with n states
    and m actions. Work with state-action pairs. Sparse matrices
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
    def __init__(self, R, Q, beta, s_indices, a_indices):
        if sp.issparse(Q):
            self.Q = Q.tocsr()
            self._sparse = True
        else:
            self.Q = np.asarray(Q)
            self._sparse = False

        self.num_sa_pairs, self.num_states = self.Q.shape

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

        _s_indices = np.empty(self.num_sa_pairs, dtype=int)
        for i in range(self.num_states):
            for j in range(self.a_indptr[i], self.a_indptr[i+1]):
                _s_indices[j] = i
        self.s_indices = _s_indices

        self.R = R[sa_ptrs.data]

        if not (0 < beta < 1):
            raise ValueError('beta must be in (0, 1)')
        self.beta = beta

        self.epsilon = 1e-3
        self.max_iter = 100
        self.tol = 1e-8

        # Linear equation solver to be used in evaluate_policy
        if self._sparse:
            self._solve = sp.linalg.spsolve
            self._I = sp.identity(self.num_states)
        else:
            self._solve = np.linalg.solve
            self._I = sp.identity(self.num_states)

    def RQ_sigma(self, sigma):
        sigma_indices = np.empty(self.num_states, dtype=int)
        _find_indices(self.a_indices, self.a_indptr, sigma, sigma_indices)

        return self.R[sigma_indices], self.Q[sigma_indices]

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
        vals = self.R + self.beta * self.Q.dot(w)  # ndim=1, size=m*n
        vals = vals.reshape(self.num_states, self.num_actions)  # n x m

        if compute_policy:
            sigma = vals.argmax(axis=1)
            Tw = vals[np.arange(self.num_states), sigma]
            return Tw, sigma
        else:
            Tw = vals.max(axis=1)
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
        # Faster than vals[np.arange(self.num_states), sigma]
        #R_sigma = self.R[np.arange(self.num_states), sigma]
        #Q_sigma = self.Q[np.arange(self.num_states), sigma]
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
        #R_sigma = self.R[np.arange(self.num_states), sigma]
        #Q_sigma = self.Q[np.arange(self.num_states), sigma]
        R_sigma, Q_sigma = self.RQ_sigma(sigma)
        b = R_sigma

        #A = np.identity(self.num_states) - self.beta * Q_sigma
        A = self._I - self.beta * Q_sigma

        #v_sigma = np.linalg.solve(A, b)
        #v_sigma = sp.linalg.spsolve(A, b)
        v_sigma = self._solve(A, b)

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
            raise ValueError

        mc = self.controlled_mc(sigma_star)

        if return_num_iter:
            return v_star, sigma_star, mc, num_iter
        else:
            return v_star, sigma_star, mc

    def successive_approx(self, T, w_0, tol, max_iter):
        # May be replaced with quantecon.compute_fixed_point
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
            #w_0 = self.R.max(axis=1)
            w_0 = self.R.reshape((self.num_states, self.num_actions)).max(axis=1)
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
            #w_0 = self.R.max(axis=1)
            w_0 = self.R.reshape((self.num_states, self.num_actions)).max(axis=1)
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
            #w_0 = self.R.max(axis=1)
            w_0 = self.R.reshape((self.num_states, self.num_actions)).max(axis=1)
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
        #P = self.Q[np.arange(self.num_states), sigma]
        _, Q_sigma = self.RQ_sigma(sigma)
        if self._sparse:
            Q_sigma = Q_sigma.toarray()
        return MarkovChain(Q_sigma)


def random_mdp_sa(num_states, num_actions, beta=None, constraints=None,
                  k=None, scale=1):
    """
    Generate an MDP randomly. The reward values are drawn from the
    normal distribution with mean 0 and standard deviation `scale`.

    Parameters
    ----------
    num_states : scalar(int)
        Number of states.

    num_actions : scalar(int)
        Number of actions.

    beta : scalar(float), optional
        Discount factor. Randomly chosen from (0, 1) if not specified.

    constraints : array_like(bool, ndim=2), optional
        Array of shape (num_states, num_actions) representing the
        constraints. If constraints[s, a] = False, then the flow reward
        of action a for state s will be set to `-inf`.

    k : scalar(int), optional
        Number of possible next states for each state-action pair. Equal
        to `num_states` if not specified.

    scale : scalar(float), optional(default=1)
        Standard deviation of the normal distribution for the reward
        values.

    Returns
    -------
    mdp_sa : MDP_sa
        An instance of MDP_sa.

    """
    if k is None:
        k = num_states

    if constraints is not None:
        s_indices, a_indices = np.where(np.asarray(constraints)==False)
        L = len(s_indices) * len(a_indices)
    else:
        from quantecon.cartesian import cartesian
        s_indices, a_indices = \
            cartesian((range(num_states), range(num_actions))).astype(int).transpose()
        L = num_states * num_actions

    R = scale * np.random.randn(L)

    rows = np.empty(L*k, dtype=int)
    for i in range(L):
        rows[k*i:k*(i+1)] = i
    cols = _random_indices(num_states, k, L).flatten()
    data = _random_probvec(k, L).flatten()
    Q = sp.coo_matrix((data, (rows, cols)), shape=(L, num_states))

    if beta is None:
        beta = np.random.rand(1)[0]

    mdp_sa = MDP_sa(R, Q, beta, s_indices, a_indices)
    return mdp_sa


@jit(nopython=True)
def _find_indices(a_indices, a_indptr, sigma, out):
    n = len(sigma)
    for i in range(n):
        for j in range(a_indptr[i], a_indptr[i+1]):
            if sigma[i] == a_indices[j]:
                out[i] = j
