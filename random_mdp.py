"""
Filename: random_mdp.py

Author: Daisuke Oyama

Generate an MDP randomly.

"""
import numpy as np
import scipy.sparse as sp
from mdp import MDP
from random_stochmatrix import gen_random_stochmatrix, \
                               _random_probvec, _random_indices


def random_mdp(num_states, num_actions, beta=None, constraints=None,
                  k=None, scale=1, sa_pair=False):
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
    mdp : MDP
        An instance of MDP.

    """
    if sa_pair:
        if k is None:
            k = num_states

        if constraints is not None:
            s_indices, a_indices = np.where(np.asarray(constraints) == False)
            L = len(s_indices) * len(a_indices)
        else:
            from quantecon.cartesian import cartesian
            s_indices, a_indices = \
                cartesian(
                    (range(num_states), range(num_actions))
                ).astype(int).transpose()
            L = num_states * num_actions

        R = scale * np.random.randn(L)

        rows = np.empty(L*k, dtype=int)
        for i in range(L):
            rows[k*i:k*(i+1)] = i
        cols = _random_indices(num_states, k, L).flatten()
        data = _random_probvec(k, L).flatten()
        Q = sp.coo_matrix((data, (rows, cols)), shape=(L, num_states))

    else:  # Not sa_pair
        R = scale * np.random.randn(num_states, num_actions)
        if constraints is not None:
            R[np.where(np.asarray(constraints) is False)] = -np.inf

        Q = np.empty((num_states, num_actions, num_states))
        Ps = gen_random_stochmatrix(num_states, k=k, num_matrices=num_actions)
        for a, P in enumerate(Ps):
            Q[:, a, :] = P

        s_indices, a_indices = None, None

    if beta is None:
        beta = np.random.rand(1)[0]

    mdp = MDP(R, Q, beta, s_indices, a_indices)
    return mdp
