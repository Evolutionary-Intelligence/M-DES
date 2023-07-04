# README

Note that all code except `dlmcma.py` are based on our [PyPop7](https://github.com/Evolutionary-Intelligence/pypop) library, which provides a unified interface to black-box optimization.

Note that `dlmcma.py` is the **core** code of our proposed meta-framework for distributed evolution strategies, where one latest variant of [LM-CMA](https://pypop.readthedocs.io/en/latest/es/lmcma.html) (called [MMES](https://pypop.readthedocs.io/en/latest/es/mmes.html)) is used as the basic computing unit (as the inner-ESs) owing to its much simpler algorithm structure than the original LM-CMA.
