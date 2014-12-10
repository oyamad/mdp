"""
Filename: mdp.py

Author: Daisuke Oyama

Base class for solving Markov decision processes (MDP) with discrete
states and actions.

"""
import numpy as np
from quantecon import MarkovChain


class MDP(object):
    """
    Class for dealing with Markov decision processes (MDP) with n states
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
        Discount factor. Must be in [0, 1).

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

        if not (0 <= beta < 1):
            raise ValueError('beta must be in [0, 1)')
        self.beta = beta

        self.tol = 1e-3
        self.max_iter = 100

    def solve(self, method='value_iteration',
              w_0=None, tol=None, max_iter=None, return_num_iter=False):
        """
        Solve the dynamic programming problem.

        """
        if method == 'value_iteration':
            v_star, num_iter = \
                self.value_iteration(w_0=w_0, tol=tol, max_iter=max_iter)
            sigma_star = self.compute_greedy(v_star)
        elif method == 'policy_iteration':
            raise NotImplementedError
        elif method == 'policy_iteration_modified':
            raise NotImplementedError
        else:
            raise ValueError

        mc = self.controlled_mc(sigma_star)

        if return_num_iter:
            return v_star, sigma_star, mc, num_iter
        else:
            return v_star, sigma_star, mc

    def value_iteration(self, w_0=None, tol=None, max_iter=None):
        if w_0 is None:
            w_0 = self.R.max(axis=1)
        if tol is None:
            tol = self.tol
        if max_iter is None:
            max_iter = self.max_iter

        w = w_0
        for i in range(max_iter):
            new_w = self.bellman_operator(w)
            if np.abs(new_w - w).max() <= tol:
                w = new_w
                break
            w = new_w

        num_iter = i + 1

        return w, num_iter

    def policy_iteration(self, sigma_0=None, max_iter=None):
        pass

    def policy_iteration_modified(self, sigma_0=None, tol=None, max_iter=None):
        pass

    def bellman_operator(self, w, return_policy=False):
        vals = self.R + self.beta * self.Q.dot(w)  # n x m

        if return_policy:
            sigma = vals.argmax(axis=1)
            return sigma
        else:
            Tw = vals.max(axis=1)
            return Tw

    def compute_greedy(self, w):
        sigma = self.bellman_operator(w, return_policy=True)
        return sigma

    def evaluate_policy(self, sigma):
        pass

    def evaluate_policy_iterative(self, sigma,
                                  w_0=None, tol=None, max_iter=None):
        pass

    def controlled_mc(self, sigma):
        """
        Returns the controlled Markov chain for a given policy `sigma`.

        """
        P = self.Q[np.arange(self.num_states), sigma]
        return MarkovChain(P)


def random_mdp(num_states, num_actions, beta=None, constraints=None):
    """
    Generate an MDP randomly.

    Parameters
    ----------
    num_states : scalar(int)
        Number of states.

    num_actions : scalar(int)
        Number of actions.

    beta : scalar(float), optional
        Discount factor. Randomly chosen from [0, 1) if not specified.

    constraints : array_like(bool, ndim=2), optional
        Array of shape (num_states, num_actions) representing the
        constraints. If constraints[s, a] = False, then the flow reward
        of action a for state s will be set to `-inf`.

    Returns
    -------
    mdp : MDP
        An instance of MDP.

    """
    R = np.random.randint(100, size=(num_states, num_actions)).astype(float)
    if constraints is not None:
        R[np.where(np.asarray(constraints) == False)] = -np.inf

    P = np.random.rand(num_states, num_actions, num_states)
    P /= np.sum(P, axis=2, keepdims=True)

    if beta is None:
        beta = np.random.rand(1)[0]

    mdp = MDP(R, P, beta)
    return mdp
