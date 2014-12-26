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
from quantecon import MarkovChain
from random_stochmatrix import gen_random_stochmatrix


class MDP(object):
    """
    Class for dealing with a Markov decision process (MDP) with n states
    and m actions.

    Parameters
    ----------
    R : array_like(ndim=2)
        Array representing the reward function, of shape n x m, where
        R[i, a] is the flow reward when the current state is i and the
        action chosen is a.

    Q : array_like(ndim=3)
        Array representing the transition probabilities, of shape
        n x m x n, where Q[i, a, j] is the probability that the state in
        the next period is j when the current state is i and the action
        chosen is a.

    beta : scalar(float), optional(default=0.95)
        Discount factor. Must be in (0, 1).

    Attributes
    ----------
    R, Q, beta : see Parameters.

    num_states : scalar(int)
        Number of states.

    num_actions : scalar(int)
        Number of actions.

    """
    def __init__(self, R, Q, beta=0.95):
        self.R, self.Q = np.asarray(R), np.asarray(Q)
        if R.ndim != 2:
            raise ValueError('R must be 2-dimenstional')
        self.num_states, self.num_actions = R.shape

        if Q.ndim != 3:
            raise ValueError('Q must be 3-dimenstional')
        if Q.shape != (self.num_states, self.num_actions, self.num_states):
            raise ValueError(
                'R and Q must be of shape n x m and n x m x n, respectively'
            )

        if not (0 < beta < 1):
            raise ValueError('beta must be in (0, 1)')
        self.beta = beta

        self.epsilon = 1e-3
        self.max_iter = 100
        self.tol = 1e-8

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
        vals = self.R + self.beta * self.Q.dot(w)  # n x m

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
        R_sigma = self.R[np.arange(self.num_states), sigma]
        Q_sigma = self.Q[np.arange(self.num_states), sigma]
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
        R_sigma = self.R[np.arange(self.num_states), sigma]
        Q_sigma = self.Q[np.arange(self.num_states), sigma]
        A = np.identity(self.num_states) - self.beta * Q_sigma
        b = R_sigma
        v_sigma = np.linalg.solve(A, b)
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
            w_0 = self.R.max(axis=1)
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
            w_0 = self.R.max(axis=1)
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
            w_0 = self.R.max(axis=1)
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
        P = self.Q[np.arange(self.num_states), sigma]
        return MarkovChain(P)


def random_mdp(num_states, num_actions, beta=None, constraints=None,
               k=None, scale=1, sparse=False):
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

    sparse : bool, optional(default=False)
        (Sparse matrices are not supported yet.)

    Returns
    -------
    mdp : MDP
        An instance of MDP.

    """
    R = scale * np.random.randn(num_states, num_actions)
    if constraints is not None:
        R[np.where(np.asarray(constraints) is False)] = -np.inf

    if sparse:
        raise NotImplementedError

    Q = np.empty((num_states, num_actions, num_states))
    Ps = gen_random_stochmatrix(num_states, k=k, num_matrices=num_actions,
                                sparse=sparse)
    for a, P in enumerate(Ps):
        Q[:, a, :] = P

    if beta is None:
        beta = np.random.rand(1)[0]

    mdp = MDP(R, Q, beta)
    return mdp
