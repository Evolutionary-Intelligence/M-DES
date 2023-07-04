import os
import time
import pickle
import argparse

import numpy as np  # for numerical computing
import pypop7  # for all benchmarking optimizers

import pypoplib.continuous_functions as cf  # for rotated and shifted benchmarking functions


class Experiment(object):
    """Each experiment consists of four settings:
        experiment index,
        test function,
        random seed,
        function dimensionality.
    """
    def __init__(self, index, function, seed, ndim_problem):
        self.index = index  # index of each experiment
        self.function = function  # test function of each experiment
        self.seed = seed  # random seed of each experiment
        self.ndim_problem = ndim_problem  # function dimensionality
        self._folder = 'pypop7_benchmarks_lso'  # folder to save all data
        if not os.path.exists(self._folder):
            os.makedirs(self._folder)
        # to set file name for each experiment
        self._file = os.path.join(self._folder, 'Algo-{}_Func-{}_Dim-{}_Exp-{}.pickle')

    def run(self, optimizer):
        # first to define all the necessary properties of the function to be minimized
        problem = {'fitness_function': self.function,
                   'ndim_problem': self.ndim_problem,
                   'upper_boundary': 10.0*np.ones((self.ndim_problem,)),
                   'lower_boundary': -10.0*np.ones((self.ndim_problem,))}
        # second to define all the necessary properties of the black-box optimizer considered
        options = {'max_function_evaluations': np.Inf,  # here we focus on the wall-clock time
                   'max_runtime': 3600*3,  # maximal runtime (seconds)
                   'fitness_threshold': 1e-10,  # fitness threshold to stop optimization
                   'seed_rng': self.seed,  # seed for random number generation
                   'saving_fitness': 2000,  # to compress the convergence data (for saving storage space)
                   'verbose': 0}  # to not print verbose information
        options['sigma'] = 20.0/3.0  # note that not all optimizers will use this setting (for ESs)
        options['temperature'] = 100  # note that not all optimizers will use this setting (for simulated annealing)
        options['n_inner_es'] = 380  # only for our meta-framework on unimodal functions (number of parallel inner-ESs)
        solver = optimizer(problem, options)  # to initialize the optimizer
        results = solver.optimize()  # to run the optimization/evolution process
        file = self._file.format(solver.__class__.__name__,
                                 solver.fitness_function.__name__,
                                 solver.ndim_problem,
                                 self.index)
        with open(file, 'wb') as handle:  # to save all data in .pickle format
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Experiments(object):
    """A set of independent experiments starting and ending in the given index range."""
    def __init__(self, start, end, ndim_problem):
        self.start = start  # starting index
        self.end = end  # ending index
        self.ndim_problem = ndim_problem  # function dimensionality
        self.indices = range(self.start, self.end + 1)  # index range (1-based rather 0-based)
        self.functions = [cf.sphere, cf.cigar, cf.discus, cf.cigar_discus, cf.ellipsoid,
                          cf.different_powers, cf.schwefel221, cf.step, cf.rosenbrock, cf.schwefel12]
        self.seeds = np.random.default_rng(2022).integers(  # to generate all random seeds in advances
            np.iinfo(np.int64).max, size=(len(self.functions), 50))

    def run(self, optimizer):
        for index in self.indices:
            print('* experiment: {:d} ***:'.format(index))
            for d, f in enumerate(self.functions):
                start_time = time.time()
                print('  * function: {:s}:'.format(f.__name__))
                experiment = Experiment(index, f, self.seeds[d, index], self.ndim_problem)
                experiment.run(optimizer)
                print('    runtime: {:7.5e}.'.format(time.time() - start_time))


if __name__ == '__main__':
    start_runtime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', '-s', type=int)  # starting index
    parser.add_argument('--end', '-e', type=int)  # ending index
    parser.add_argument('--optimizer', '-o', type=str)  # optimizer
    parser.add_argument('--ndim_problem', '-d', type=int, default=2000)  # dimension (default to 2000)
    args = parser.parse_args()
    params = vars(args)
    if params['optimizer'] == 'PRS':
        from pypop7.optimizers.rs.prs import PRS as Optimizer
    elif params['optimizer'] == 'RHC':
        from pypop7.optimizers.rs.rhc import RHC as Optimizer
    elif params['optimizer'] == 'ARHC':
        from pypop7.optimizers.rs.arhc import ARHC as Optimizer
    elif params['optimizer'] == 'SRS':
        from pypop7.optimizers.rs.srs import SRS as Optimizer
    elif params['optimizer'] == 'BES':
        from pypop7.optimizers.rs.bes import BES as Optimizer
    elif params['optimizer'] == 'GENITOR':
        from pypop7.optimizers.ga.genitor import GENITOR as Optimizer
    elif params['optimizer'] == 'G3PCX':
        from pypop7.optimizers.ga.g3pcx import G3PCX as Optimizer
    elif params['optimizer'] == 'GL25':
        from pypop7.optimizers.ga.gl25 import GL25 as Optimizer
    elif params['optimizer'] == 'CSA':
        from pypop7.optimizers.sa.csa import CSA as Optimizer
    elif params['optimizer'] == 'ESA':
        from pypop7.optimizers.sa.esa.esa import ESA as Optimizer
    elif params['optimizer'] == 'NSA':
        from pypop7.optimizers.sa.nsa import NSA as Optimizer
    elif params['optimizer'] == 'COEA':
        from pypop7.optimizers.cc.coea import COEA as Optimizer
    elif params['optimizer'] == 'COSYNE':
        from pypop7.optimizers.cc.cosyne import COSYNE as Optimizer
    elif params['optimizer'] == 'COCMA':
        from pypop7.optimizers.cc.cocma import COCMA as Optimizer
    elif params['optimizer'] == 'HCC':
        from pypop7.optimizers.cc.hcc import HCC as Optimizer
    elif params['optimizer'] == 'SPSO':
        from pypop7.optimizers.pso.spso import SPSO as Optimizer
    elif params['optimizer'] == 'SPSOL':
        from pypop7.optimizers.pso.spsol import SPSOL as Optimizer
    elif params['optimizer'] == 'CPSO':
        from pypop7.optimizers.pso.cpso import CPSO as Optimizer
    elif params['optimizer'] == 'CLPSO':
        from pypop7.optimizers.pso.clpso import CLPSO as Optimizer
    elif params['optimizer'] == 'CCPSO2':
        from pypop7.optimizers.pso.ccpso2 import CCPSO2 as Optimizer
    elif params['optimizer'] == 'CDE':
        from pypop7.optimizers.de.cde import CDE as Optimizer
    elif params['optimizer'] == 'TDE':
        from pypop7.optimizers.de.tde import TDE as Optimizer
    elif params['optimizer'] == 'JADE':
        from pypop7.optimizers.de.jade import JADE as Optimizer
    elif params['optimizer'] == 'CODE':
        from pypop7.optimizers.de.code import CODE as Optimizer
    elif params['optimizer'] == 'SCEM':
        from pypop7.optimizers.cem.scem import SCEM as Optimizer
    elif params['optimizer'] == 'DSCEM':
        from pypop7.optimizers.cem.dscem import DSCEM as Optimizer
    elif params['optimizer'] == 'MRAS':
        from pypop7.optimizers.cem.mras import MRAS as Optimizer
    elif params['optimizer'] == 'DCEM':
        from pypop7.optimizers.cem.dcem import DCEM as Optimizer
    elif params['optimizer'] == 'UMDA':
        from pypop7.optimizers.eda.umda import UMDA as Optimizer
    elif params['optimizer'] == 'EMNA':
        from pypop7.optimizers.eda.emna import EMNA as Optimizer
    elif params['optimizer'] == 'AEMNA':
        from pypop7.optimizers.eda.aemna import AEMNA as Optimizer
    elif params['optimizer'] == 'RPEDA':
        from pypop7.optimizers.eda.rpeda import RPEDA as Optimizer
    elif params['optimizer'] == 'SGES':
        from pypop7.optimizers.nes.sges import SGES as Optimizer
    elif params['optimizer'] == 'XNES':
        from pypop7.optimizers.nes.xnes import XNES as Optimizer
    elif params['optimizer'] == 'SNES':
        from pypop7.optimizers.nes.snes import SNES as Optimizer
    elif params['optimizer'] == 'R1NES':
        from pypop7.optimizers.nes.r1nes import R1NES as Optimizer
    elif params['optimizer'] == 'DLMCMA':  # Distributed
        from pypoplib.dlmcma import DLMCMA as Optimizer
    elif params['optimizer'] == 'MMES':  # 2021
        from pypop7.optimizers.es.mmes import MMES as Optimizer
    elif params['optimizer'] == 'SAMAES':  # 2020
        from pypop7.optimizers.es.samaes import SAMAES as Optimizer
    elif params['optimizer'] == 'SAES':  # 2020
        from pypop7.optimizers.es.saes import SAES as Optimizer
    elif params['optimizer'] == 'DDCMA':  # 2020
        from pypop7.optimizers.es.ddcma import DDCMA as Optimizer
    elif params['optimizer'] == 'FCMAES':  # 2020
        from pypop7.optimizers.es.fcmaes import FCMAES as Optimizer
    elif params['optimizer'] == 'LMMAES':  # 2019
        from pypop7.optimizers.es.lmmaes import LMMAES as Optimizer
    elif params['optimizer'] == 'RMES':  # 2018
        from pypop7.optimizers.es.rmes import RMES as Optimizer
    elif params['optimizer'] == 'R1ES':  # 2018
        from pypop7.optimizers.es.r1es import R1ES as Optimizer
    elif params['optimizer'] == 'FMAES':  # 2017
        from pypop7.optimizers.es.fmaes import FMAES as Optimizer
    elif params['optimizer'] == 'MAES':  # 2017
        from pypop7.optimizers.es.maes import MAES as Optimizer
    elif params['optimizer'] == 'LMCMA':  # 2017
        from pypop7.optimizers.es.lmcma import LMCMA as Optimizer
    elif params['optimizer'] == 'CCMAES2016':  # 2016
        from pypop7.optimizers.es.ccmaes2016 import CCMAES2016 as Optimizer
    elif params['optimizer'] == 'VKDCMA':  # 2016
        from pypop7.optimizers.es.vkdcma import VKDCMA as Optimizer
    elif params['optimizer'] == 'CMAES':  # 2016
        from pypop7.optimizers.es.cmaes import CMAES as Optimizer
    elif params['optimizer'] == 'OPOA2015':  # 2015
        from pypop7.optimizers.es.opoa2015 import OPOA2015 as Optimizer
    elif params['optimizer'] == 'LMCMAES':  # 2014
        from pypop7.optimizers.es.lmcmaes import LMCMAES as Optimizer
    elif params['optimizer'] == 'VDCMA':  # 2014
        from pypop7.optimizers.es.vdcma import VDCMA as Optimizer
    elif params['optimizer'] == 'OPOA2010':  # 2010
        from pypop7.optimizers.es.opoa2010 import OPOA2010 as Optimizer
    elif params['optimizer'] == 'CCMAES2009':  # 2009
        from pypop7.optimizers.es.ccmaes2009 import CCMAES2009 as Optimizer
    elif params['optimizer'] == 'OPOC2009':  # 2009
        from pypop7.optimizers.es.opoc2009 import OPOC2009 as Optimizer
    elif params['optimizer'] == 'SEPCMAES':  # 2008
        from pypop7.optimizers.es.sepcmaes import SEPCMAES as Optimizer
    elif params['optimizer'] == 'OPOC2006':  # 2006
        from pypop7.optimizers.es.opoc2006 import OPOC2006 as Optimizer
    elif params['optimizer'] == 'CSAES':  # 1994
        from pypop7.optimizers.es.csaes import CSAES as Optimizer
    elif params['optimizer'] == 'DSAES':  # 1994
        from pypop7.optimizers.es.dsaes import DSAES as Optimizer
    elif params['optimizer'] == 'RES':  # 1973
        from pypop7.optimizers.es.res import RES as Optimizer
    experiments = Experiments(params['start'], params['end'], params['ndim_problem'])
    experiments.run(Optimizer)
    print('*** Total runtime: {:7.5e} ***.'.format(time.time() - start_runtime))
