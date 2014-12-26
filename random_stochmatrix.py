"""
Filename: random_stochmatrix.py

Author: Daisuke Oyama

May be put in mc_tools.py

"""
import numpy as np
import scipy.sparse
from numba import jit


def gen_random_stochmatrix(n, k=None, num_matrices=1,
                           sparse=False, format='csr'):
    """
    Generate random stochastic matrices.

    Parameters
    ----------
    n : scalar(int)
        Number of states for each stochastic matrix.

    k : scalar(int), optional
        Number of nonzero entries in each row of each matrix. Set to n
        if not specified.

    num_matrices : scalar(int), optional(default=1)
        Number of matrices to generate.

    sparse : bool, optional(default=False)
        Whether to generate matrices in sparse matrix form.

    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'},
             optional(default='csr')
        Sparse matrix format. Relevant only when sparse=True.

    Returns
    -------
    generator of numpy ndarrays or scipy sparse matrices (float, ndim=2)
        Stochastic matrices.

    """
    if k is None:
        k = n
    if not (isinstance(k, int) and 0 < k <= n):
        raise ValueError('k must be an integer with 0 < k <= n')

    x = np.empty((num_matrices, n, k+1))
    r = np.random.rand(num_matrices, n, k-1)
    r.sort(axis=-1)
    x[:, :, 0], x[:, :, 1:k], x[:, :, k] = 0, r, 1
    probs = np.diff(x, axis=-1)

    if k == n:
        for m in range(num_matrices):
            if sparse:
                yield scipy.sparse.coo_matrix(probs[m]).asformat(format)
            else:
                yield probs[m]

    else:
        rows = np.empty(n*k, dtype=int)
        for i in range(n):
            rows[k*i:k*(i+1)] = i
        for m in range(num_matrices):
            cols = _random_indices(n, k, n).flatten()
            data = probs[m].flatten()

            if sparse:
                P = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
                yield P.asformat(format)
            else:
                P = np.zeros((n, n))
                P[rows, cols] = data
                yield P

@jit
def _random_indices(n, k, m):
    """
    Create m arrays of k integers randomly chosen without replacement
    from 0, ..., n-1. About 10x faster than numpy.random.choice with
    replace=False. Logic taken from random.sample.

    Parameters
    ----------
    n : scalar(int)
        Number of integers, 0, ..., n-1, to sample from.

    k : scalar(int)
        Number of elements of each array.

    m : scalar(int)
        Number of arrays.

    Returns
    -------
    result : ndarray(int, ndim=2)
        m x k array. Each row contains k unique integers chosen from
        0, ..., n-1.

    """
    r = np.random.rand(m, k)

    result = np.empty((m, k), dtype=int)
    pool = np.empty((m, n), dtype=int)
    for i in range(m):
        for j in range(n):
            pool[i, j] = j

    for i in range(m):
        for j in range(k):
            idx = np.floor(r[i, j] * (n-j))
            result[i, j] = pool[i, idx]
            pool[i, idx] = pool[i, n-j-1]

    return result
