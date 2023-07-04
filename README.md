# M-DES (Meta-framework for Distributed Evolution Strategies)

This is the companion website of the paper "Distributed LM-CMA with Multi-Level Learning for Large-Scale Black-Box Optimization", submitted to [IEEE-TEVC](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4235) (**under review**). To ensure *repeatability* and promote *benchmarking*, all data and code involved in this paper are given here.

For all benchmarking optimizers, their source code are openly accessed via our recently-designed Python libary called [PyPop7](https://github.com/Evolutionary-Intelligence/pypop). For our **meta-framework for distributed evolution strategies** proposed in this paper, its source code is located in the folder [pypoplib](https://github.com/Evolutionary-Intelligence/M-DES/tree/main/pypoplib).

# Project Structure

* **README.md**: for basic information of the companion website
* **run_experiments.py**: script to run all numerical experiments for all black-box optimizers
* **plot_median_vs_es.py**: to print (median) convergence curves when compared with [ESs](https://pypop.readthedocs.io/en/latest/es/es.html)
* **plot_median_vs_others.py**: to print (median) convergence curves when compared with [all others](https://pypop.readthedocs.io/en/latest/index.html) (except ESs)
