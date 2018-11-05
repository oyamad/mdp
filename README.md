mdp
===

This module has been merged in
[QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py)
(version 0.2.0 or above) as
[`DiscreteDP`](https://quanteconpy.readthedocs.io/en/latest/markov/ddp.html).

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


## Lecture in [Quantitative Economics](https://lectures.quantecon.org)

* [Discrete State Dynamic Programming](https://lectures.quantecon.org/py/discrete_dp.html)


## Notebooks

* [Getting started](http://notes.quantecon.org/submission/5bd7a72df966080015bafbd1)
* [Implementation details](http://notes.quantecon.org/submission/5bd7a7c2f966080015bafbd2)
* Examples
  * [Automobile replacement (Rust 1996)](http://notes.quantecon.org/submission/5b3585efb9eab00015b89f87)
  * [Optimal growth](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/ddp_ex_optgrowth_py.ipynb)
  * [Job search](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/ddp_ex_job_search_py.ipynb)
  * [Career choice](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/ddp_ex_career_py.ipynb)
  * [Mine Managment (Miranda and Fackler Section 7.6.1)](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/ddp_ex_MF_7_6_1_py.ipynb)
  * [Asset replacement (Miranda and Fackler Section 7.6.2)](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/ddp_ex_MF_7_6_2_py.ipynb)
  * [Asset replacement with maintenance (Miranda and Fackler Section 7.6.3)](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/ddp_ex_MF_7_6_3_py.ipynb)
  * [Option pricing (Miranda and Fackler Section 7.6.4)](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/ddp_ex_MF_7_6_4_py.ipynb)
  * [Water management (Miranda and Fackler Section 7.6.5)](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/ddp_ex_MF_7_6_5_py.ipynb)
  * [Bioeconomics (Miranda and Fackler Section 7.6.6)](http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/ddp_ex_MF_7_6_6_py.ipynb)
  * [POMDP Tiger example](http://nbviewer.jupyter.org/github/oyamad/mdp/blob/master/pomdp_tiger.ipynb)
* Perfomance comparison
  * [Machine 1](http://nbviewer.jupyter.org/github/oyamad/mdp/blob/master/ddp_performance.ipynb)
  * [Machine 2](http://nbviewer.jupyter.org/github/oyamad/mdp/blob/master/ddp_performance-2.ipynb)
