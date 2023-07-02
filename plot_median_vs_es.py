"""This Python script only plots the *median* convergence curves for
    all black-box optimizers (BBO) considered: ES.

    https://github.com/Evolutionary-Intelligence/pypop
    https://pypop.readthedocs.io/en/latest/index.html
"""
import os
import sys
import pickle  # for data storage

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams

from pypoplib import optimizer
sys.modules['optimizer'] = optimizer  # for `pickle`


def read_pickle(s, f, i):
    data_folder = 'pypop7_benchmarks_lso'
    afile = os.path.join(data_folder, 'Algo-' + s + '_Func-' + f + '_Dim-2000_Exp-' + i + '.pickle')
    with open(afile, 'rb') as handle:
        return pickle.load(handle)


if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '12'

    n_trials = 10
    algos = ['LMCMA', 'MMES', 'FCMAES', 'DDCMA', 'LMMAES',
        'RMES', 'R1ES', 'VKDCMA', 'VDCMA', 'LMCMAES',
        'FMAES', 'MAES', 'CCMAES2016', 'OPOA2015', 'OPOA2010',
        'CCMAES2009', 'OPOC2009', 'SEPCMAES', 'OPOC2006', 'CMAES',
        'SAMAES', 'SAES', 'CSAES', 'DSAES', 'RES',
        'DLMCMA']
    max_runtime, fitness_threshold = 3600*3 - 10*60, 1e-10
    colors, c = [name for name, _ in colors.cnames.items()], []
    for i in [7] + list(range(9, 18)) + list(range(19, len(colors))):
        c.append(colors[i])  # for better colors
    funcs = ['sphere', 'cigar', 'discus', 'cigar_discus', 'ellipsoid',
        'different_powers', 'schwefel221', 'step', 'rosenbrock', 'schwefel12']
    for k, f in enumerate(funcs):
        print('* {:s} ***'.format(f))
        time, fitness = [], []
        for j in range(len(algos)):  # initialize
            time.append([])
            fitness.append([])
            for i in range(n_trials):
                time[j].append([])
                fitness[j].append([])
        for i in range(n_trials):
            b = []
            for j, a in enumerate(algos):
                results = read_pickle(a, f, str(i + 1))
                b.append(results['best_so_far_y'])
                time[j][i] = results['fitness'][:, 0]*results['runtime']/results['n_function_evaluations']
                y = results['fitness'][:, 1]
                for i_y in range(1, len(y)):  # for best-so-far fitness curve
                    if y[i_y] > y[i_y - 1]:
                         y[i_y] = y[i_y - 1]
                fitness[j][i] = y
            for j, b in enumerate(b):
                print('{:s}: {:5.2e} '.format(algos[j], b), end='')
            print()
        plt.figure(figsize=(5, 5))
        plt.yscale('log')
        top_ranked = []
        for j, a in enumerate(algos):
            end_runtime = [time[j][i][-1] for i in range(len(time[j]))]
            end_fit = [fitness[j][i][-1] for i in range(len(fitness[j]))]
            order = np.argsort(end_runtime)[int(n_trials/2)]  # for median (but non-standard)
            _r = end_runtime[order] if end_runtime[order] <= max_runtime else max_runtime
            _f = end_fit[order] if end_fit[order] >= fitness_threshold else fitness_threshold 
            top_ranked.append([_r, _f, a])
        top_ranked.sort(key= lambda x: (x[0], x[1]))  # sort by first runtime then fitness
        top_ranked = [t for t in [tr[2] for tr in top_ranked]]
        print('  #top:', top_ranked)
        for j, a in enumerate(algos):
            end_runtime = [time[j][i][-1] for i in range(len(time[j]))]
            order = np.argsort(end_runtime)[int(n_trials/2)]  # for median (but non-standard)
            if a in top_ranked:
                plt.plot(time[j][order], fitness[j][order], label=a, color=c[j])
            else:
                plt.plot(time[j][order], fitness[j][order], color=c[j])
        plt.xlabel('Running Time (Seconds)')
        plt.ylabel('Fitness')
        plt.title(f)
        # plt.legend(loc=4, ncol=6)
        plt.show()
