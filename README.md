# M-DES (Meta-framework for Distributed Evolution Strategies)

This is the companion website of the paper "Distributed LM-CMA with Multi-Level Learning for Large-Scale Black-Box Optimization", submitted to [IEEE-TPDS](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4235) (**under review**). To ensure *repeatability* and promote *benchmarking*, all data and code involved in this paper are given here.

For all benchmarking optimizers, their source code are openly accessed via our recently-designed Python libary called [PyPop7](https://github.com/Evolutionary-Intelligence/pypop). [PyPop7 is a Pure-PYthon library of POPulation-based OPtimization for single-objective, real-parameter, black-box problems](https://pypop.readthedocs.io/). For our **meta-framework for distributed evolution strategies** proposed in this paper, its source code is located in the folder [pypoplib](https://github.com/Evolutionary-Intelligence/M-DES/tree/main/pypoplib).

# Project Structure

* **README.md**: basic information of the companion website,
* **run_experiments.py**: script to run all numerical experiments for all black-box optimizers (BBO),
* **plot_median_vs_es.py**: to print (median) convergence curves when compared with [ESs](https://pypop.readthedocs.io/en/latest/es/es.html),
* **plot_median_vs_others.py**: to print (median) convergence curves when compared with [all others](https://pypop.readthedocs.io/en/latest/index.html) (except ESs),
* **data**: to store all data generated in numerical experiments,
* **figures**: to store all figures generated from data,
* **pypoplib**: source code for our proposed meta-framework and also (rotated-shifted) benchmarking functions.

# Bash for Python Virtual Environment (Conda)

* First install the software [miniconda](https://docs.conda.io/en/latest/miniconda.html) on all Linux servers (*suggested*) and then run the following `bash` commands:

```bash
$ conda deactivate
$ mkdir PyProjects  # optional
$ cd PyProjects/  # optional
$ mkdir tpds2022  # optional
$ cd tpds2022/  # optional
$ conda create -y --prefix env_tpds2022
$ conda activate env_tpds2022/
$ conda install -y --prefix env_tpds2022/ python=3.8.12
$ pip install numpy==1.24.2
$ pip install scipy==1.10.1
$ pip install scikit-learn==1.2.2
$ pip install ray==2.3.1
$ pip install pypop7
$ python
>>> from importlib.metadata import version
>>> version('numpy')
>>> version('scipy')
>>> version('scikit-learn')
>>> version('ray')
>>> version('pypop7')
```
