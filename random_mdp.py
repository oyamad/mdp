"""
Filename: random_mdp.py

Author: Daisuke Oyama

Generate an MDP randomly.

"""
import numpy as np
import scipy.sparse
from numba import jit
from quantecon.random import probvec, sample_without_replacement
from quantecon.util import check_random_state
from mdp import MDP


def _random_stochastic_matrix(m, n, k=None, sparse=False, format='csr',
                              random_state=None):
    if k is None:
        k = n
    # m prob vectors of dimension k, shape (m, k)
    probvecs = probvec(m, k, random_state=random_state)

    if k == n:
        P = probvecs
        if sparse:
            return scipy.sparse.coo_matrix(P).asformat(format)
        else:
            return P

    # if k < n:
    rows = np.repeat(np.arange(m), k)
    cols = \
        sample_without_replacement(
            n, k, num_trials=m, random_state=random_state
        ).ravel()
    data = probvecs.ravel()

    if sparse:
        P = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(m, n))
        return P.asformat(format)
    else:
        P = np.zeros((m, n))
        P[rows, cols] = data
        return P


def random_mdp(num_states, num_actions, beta=None,
               k=None, scale=1, sparse=False, sa_pair=False,
               random_state=None):
    """
    Generate an MDP randomly. The reward values are drawn from the
    normal distribution with mean 0 and standard deviation `scale`.

    Parameters
    ----------
    num_states : scalar(int)
        Number of states.

    num_actions : scalar(int)
        Number of actions.

    beta : scalar(float), optional(default=None)
        Discount factor. Randomly chosen from [0, 1) if not specified.

    k : scalar(int), optional(default=None)
        Number of possible next states for each state-action pair. Equal
        to `num_states` if not specified.

    scale : scalar(float), optional(default=1)
        Standard deviation of the normal distribution for the reward
        values.

    sparse : bool, optional(default=False)
        Whether to store the transition probability array in sparse
        matrix form.

    sa_pair : bool, optional(default=False)
        Whether to represent the data in the state-action pairs
        formulation. (If `sparse=True`, automatically set `True`.)

    random_state : scalar(int) or np.random.RandomState,
                   optional(default=None)
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    mdp : MDP
        An instance of MDP.

    """
    if sparse:
        sa_pair = True

    # Number of state-action pairs
    L = num_states * num_actions

    random_state = check_random_state(random_state)
    R = scale * random_state.randn(L)
    Q = _random_stochastic_matrix(L, num_states, k=k,
                                  sparse=sparse, format='csr',
                                  random_state=random_state)
    if beta is None:
        beta = random_state.random_sample()

    if sa_pair:
        s_indices, a_indices = _sa_indices(num_states, num_actions)
    else:
        s_indices, a_indices = None, None
        R.shape = (num_states, num_actions)
        Q.shape = (num_states, num_actions, num_states)

    mdp = MDP(R, Q, beta, s_indices, a_indices)
    return mdp


@jit
def _sa_indices(num_states, num_actions):
    L = num_states * num_actions
    s_indices = np.empty(L, dtype=int)
    a_indices = np.empty(L, dtype=int)

    i = 0
    for s in range(num_states):
        for a in range(num_actions):
            s_indices[i] = s
            a_indices[i] = a
            i += 1

    return s_indices, a_indices
