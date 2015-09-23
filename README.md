mdp
===

This module has been merged in
[QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py)
(version 0.2.0 or above) as
[`DiscreteDP`](https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/markov/ddp.py).

To try, type

```
pip install quantecon
```

at a terminal prompt.


## Quick Start

```python
from quantecon.markov import DiscreteDP
```

**Creating a `DiscreteDP` instance**

*Product formulation*

```python
>>> R = [[5, 10], [-1, -float('inf')]]
>>> Q = [[(0.5, 0.5), (0, 1)], [(0, 1), (0.5, 0.5)]]
>>> beta = 0.95
>>> ddp = DiscreteDP(R, Q, beta)
```

*State-action pairs formulation*

```python
>>> s_indices = [0, 0, 1]  # State indices
>>> a_indices = [0, 1, 0]  # Action indices
>>> R = [5, 10, -1]
>>> Q = [(0.5, 0.5), (0, 1), (0, 1)]
>>> beta = 0.95
>>> ddp = DiscreteDP(R, Q, beta, s_indices, a_indices)
```

**Solving the model**

*Policy iteration*

```python
>>> res = ddp.solve(method='policy_iteration', v_init=[0, 0])
>>> res.sigma  # Optimal policy function
array([0, 0])
>>> res.v  # Optimal value function
array([ -8.57142857, -20.        ])
>>> res.num_iter  # Number of iterations
2
```

*Value iteration*

```python
>>> res = ddp.solve(method='value_iteration', v_init=[0, 0],
...                 epsilon=0.01)
>>> res.sigma  # (Approximate) optimal policy function
array([0, 0])
>>> res.v  # (Approximate) optimal value function
array([ -8.5665053 , -19.99507673])
>>> res.num_iter  # Number of iterations
162
```

*Modified policy iteration*

```python
>>> res = ddp.solve(method='modified_policy_iteration',
...                 v_init=[0, 0], epsilon=0.01)
>>> res.sigma  # (Approximate) optimal policy function
array([0, 0])
>>> res.v  # (Approximate) optimal value function
array([ -8.57142826, -19.99999965])
>>> res.num_iter  # Number of iterations
3
```


## Lecture in [quant-econ.net](http://quant-econ.net)

* [Discrete Dynamic Programming](http://quant-econ.net/py/discrete_dp.html)


## Notebooks

* [Getting started](http://nbviewer.ipython.org/github/QuantEcon/QuantEcon.site/blob/mdp/_static/notebooks/inwork/ddp/ddp_intro.ipynb)
* [Implementation details](http://nbviewer.ipython.org/github/QuantEcon/QuantEcon.site/blob/mdp/_static/notebooks/inwork/ddp/ddp_theory.ipynb)
* Examples
  * [Automobile replacement (Rust 1996)](http://nbviewer.ipython.org/github/QuantEcon/QuantEcon.site/blob/mdp/_static/notebooks/inwork/ddp/ddp_ex_rust96.ipynb)
  * [Optimal growth](http://nbviewer.ipython.org/github/oyamad/mdp/blob/master/ddp_ex_optgrowth.ipynb)
  * [Job search](http://nbviewer.ipython.org/github/oyamad/mdp/blob/master/ddp_ex_job_search.ipynb)
  * [Career choice](http://nbviewer.ipython.org/github/oyamad/mdp/blob/master/ddp_ex_career.ipynb)
  * [Asset replacement (Miranda and Fackler Section 7.6.2)](http://nbviewer.ipython.org/github/oyamad/mdp/blob/master/ddp_ex_MF_7_6_2.ipynb)
  * [Asset replacement with maintenance (Miranda and Fackler Section 7.6.3)](http://nbviewer.ipython.org/github/oyamad/mdp/blob/master/ddp_ex_MF_7_6_3.ipynb)
  * [Option pricing (Miranda and Fackler Section 7.6.4)](http://nbviewer.ipython.org/github/oyamad/mdp/blob/master/ddp_ex_MF_7_6_4.ipynb)
  * [Water management (Miranda and Fackler Section 7.6.5)](http://nbviewer.ipython.org/github/oyamad/mdp/blob/master/ddp_ex_MF_7_6_5.ipynb)
  * [POMDP Tiger example](http://nbviewer.ipython.org/github/oyamad/mdp/blob/master/pomdp_tiger.ipynb)
* [Perfomance comparison](http://nbviewer.ipython.org/github/oyamad/mdp/blob/master/ddp_performance.ipynb)
