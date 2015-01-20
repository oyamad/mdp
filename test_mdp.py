"""
Filename: test_mdp.py
Author: Daisuke Oyama

Tests for mdp.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from nose.tools import eq_, ok_

from mdp import MDP


class TestMDP:
    def setUp(self):
        # From Puterman 2005, Section 3.1
        n, m = 2, 2  # number of states, number of actions
        R = [[5, 10], [-1, -np.inf]]
        Q = np.empty((n, m, n))
        Q[0, 0, :] = 0.5, 0.5
        Q[0, 1, :] = 0, 1
        Q[1, :, :] = 0, 1
        beta = 0.95

        self.mdp = MDP(R, Q, beta)
        self.mdp.max_iter = 200

        self.epsilon = 1e-2

        # Analytical solution for beta < 10/11, Example 6.2.1
        self.v_star = [(5-5.5*beta)/((1-0.5*beta)*(1-beta)), -1/(1-beta)]
        self.sigma_star = [0, 0]

    def test_value_iteration(self):
        v, sigma, mc = self.mdp.solve(method='value_iteration',
                                      epsilon=self.epsilon)

        # Check v is an epsilon/2-approxmation of v_star
        ok_(np.abs(v - self.v_star).max() < self.epsilon/2)

        # Check sigma == sigma_star
        assert_array_equal(sigma, self.sigma_star)

    def test_policy_iteration(self):
        w_0 = [0, 1]  # Let it iterate more than once
        v, sigma, mc = self.mdp.solve(method='policy_iteration', w_0=w_0)

        # Check v == v_star
        assert_allclose(v, self.v_star)

        # Check sigma == sigma_star
        assert_array_equal(sigma, self.sigma_star)

    def test_modified_policy_iteration(self):
        k = 5
        v, sigma, mc = self.mdp.solve(method='modified_policy_iteration',
                                      epsilon=self.epsilon, k=k)

        # Check v is an epsilon/2-approxmation of v_star
        ok_(np.abs(v - self.v_star).max() < self.epsilon/2)

        # Check sigma == sigma_star
        assert_array_equal(sigma, self.sigma_star)

    def test_modified_policy_iteration_k1(self):
        k = 1
        v, sigma, mc = self.mdp.solve(method='modified_policy_iteration',
                                      epsilon=self.epsilon, k=k)

        # Check v is an epsilon/2-approxmation of v_star
        ok_(np.abs(v - self.v_star).max() < self.epsilon/2)

        # Check sigma == sigma_star
        assert_array_equal(sigma, self.sigma_star)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
