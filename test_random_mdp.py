"""
Filename: test_random_mdp.py
Author: Daisuke Oyama

Tests for random_mdp.py

"""
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_

from random_mdp import random_mdp


class TestRandomMDP:
    def setUp(self):
        self.num_states, self.num_actions = 5, 4
        self.num_sa = self.num_states * self.num_actions
        self.k = 3
        seed = 1234

        self.mdp = random_mdp(self.num_states, self.num_actions, k=self.k,
                              sparse=False, sa_pair=False, random_state=seed)

        labels = ['dense', 'sparse']
        self.mdps_sa = {}
        for label in labels:
            is_sparse = (label == 'sparse')
            self.mdps_sa[label] = \
                random_mdp(self.num_states, self.num_actions, k=self.k,
                           sparse=is_sparse, sa_pair=True, random_state=seed)

    def test_shape(self):
        n, m, L = self.num_states, self.num_actions, self.num_sa

        eq_(self.mdp.R.shape, (n, m))
        eq_(self.mdp.Q.shape, (n, m, n))

        for mdp in self.mdps_sa.itervalues():
            eq_(mdp.R.shape, (L,))
            eq_(mdp.Q.shape, (L, n))

    def test_nonzero(self):
        n, m, L, k = self.num_states, self.num_actions, self.num_sa, self.k

        assert_array_equal((self.mdp.Q > 0).sum(axis=-1), np.ones((n, m))*k)
        assert_array_equal((self.mdps_sa['dense'].Q > 0).sum(axis=-1),
                           np.ones(L)*k)
        assert_array_equal(self.mdps_sa['sparse'].Q.getnnz(axis=-1),
                           np.ones(L)*k)

    def test_equal_reward(self):
        assert_array_equal(self.mdp.R.ravel(), self.mdps_sa['dense'].R)
        assert_array_equal(self.mdps_sa['dense'].R, self.mdps_sa['sparse'].R)

    def test_equal_probability(self):
        assert_array_equal(self.mdp.Q.ravel(), self.mdps_sa['dense'].Q.ravel())
        assert_array_equal(self.mdps_sa['dense'].Q,
                           self.mdps_sa['sparse'].Q.toarray())


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
