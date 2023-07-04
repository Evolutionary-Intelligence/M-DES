import time

import numpy as np  # engine for numerical computing
import ray  # engine for distributed computing

# only for large data (e.g., rotation matrix) involved in fitness evaluations
from pypoplib.continuous_functions import load_shift_and_rotation as load_sr
from pypoplib.es import ES  # abstract class for `ES`
from pypoplib.mmes import MMES as LMCMA  # a latest variant of Limited-Memory Covariance Matrix Adaptation


class DLMCMA(ES):  # in `pypoplib` folder
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        self.n_inner_es = options.get('n_inner_es')  # total number of inner ESs in parallel
        assert self.n_inner_es is not None and self.n_inner_es >= 40,\
            'there are at least 40 (logical) CPU cores.'
        self.runtime_inner_es = options.get('runtime_inner_es', 2.5*60)  # runtime of each inner ES (seconds) in each cycle
        assert self.runtime_inner_es >= 3  # seconds
        # to make sure actual runtime does not exceed given 'max_runtime' AMAP
        self.max_runtime -= self.runtime_inner_es  # this may result in slightly less runtime (unfair for `DES`)
        self.n_outer = int(self.n_inner_es/5)  # number of inner ESs for outer ES (MetaES)
        w_base, w = np.log((self.n_outer*2 + 1.0)/2.0), np.log(np.arange(self.n_outer) + 1.0)
        self._w_outer = (w_base - w)/(self.n_outer*w_base - np.sum(w))
        self._ww_outer = (self._w_outer)/np.sqrt(np.sum(np.square(self._w_outer)))
        self.sigma = np.max(self.upper_boundary - self.lower_boundary)/2.0
        self.m_max = 2*int(np.ceil(np.sqrt(self.ndim_problem)))
        self.m_list = np.arange(1, self.m_max) + 1
        self.p_list = 1.0/len(self.m_list)*np.ones((len(self.m_list)))

    def optimize(self, fitness_function=None):  # for all iterations (generations)
        super(ES, self).optimize(fitness_function)
        fitness = []  # to store all fitness generated during search
        # https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html
        ray.init(address='auto',  # to assume the ray clustering computing platform is available
            runtime_env={'py_modules': ['./pypoplib'],  # this local folder is shared across all nodes
                'env_vars': {'OPENBLAS_NUM_THREADS': '1',  # to close *multi-thread* for avoiding possible conflicts
                    'MKL_NUM_THREADS': '1',  # to close *multi-thread* for avoiding possible conflicts
                    'OMP_NUM_THREADS': '1',  # to close *multi-thread* for avoiding possible conflicts
                    'NUMEXPR_NUM_THREADS': '1',  # to close *multi-thread* for avoiding possible conflicts
                    'RAY_memory_monitor_refresh_ms': '0'}})  # to avoid Out-Of-Memory Prevention
        ray_problem = ray.put(self.problem)  # to be shared across all nodes
        ray_opt = ray.remote(num_cpus=1)(LMCMA)  # to be shared across all nodes
        # to avoid repeated coping and communication of the same data over network,
        #   use *ray.put* to upload them to the shared memory in each node only once
        sv, rm = load_sr(self.fitness_function, np.empty((self.ndim_problem,)))
        ray_args = ray.put({'shift_vector': sv, 'rotation_matrix': rm})
        is_first_generation = True  # flag to mark the first generation
        x, xx = self.rng_optimization.uniform(self.lower_boundary, self.upper_boundary,
            size=(self.n_inner_es, self.ndim_problem)), None  # to save the best-so-far solutions from all inner ESs
        y = np.empty((self.n_inner_es,))  # to save the best-so-far fitness from all inner ESs
        p, pp = np.zeros((self.n_inner_es, self.ndim_problem)), None
        w, ww = np.zeros((self.n_inner_es,)), None
        q, qq = np.zeros((self.n_inner_es, self.m_max, self.ndim_problem)), None
        s, ss = np.empty((self.n_inner_es,)), None
        m, mm = np.empty((self.n_inner_es,)), None
        options = [None]*self.n_inner_es
        while not self._check_terminations():
            ray_es, ray_results = [], []
            for i in range(self.n_inner_es):  # to run in parallel (driven by the engine of ray)
                if is_first_generation:
                    m_ray = int(self.rng_optimization.choice(self.m_list, p=self.p_list))
                    mean_ray, p_ray, w_ray, q_ray = x[i], p[i], w[i], q[i][-m_ray:]
                    s_ray = self.rng_optimization.uniform(1e-16, 1e-15 + self.sigma)
                else:
                    # to use *elitist* to avoid regression/stagnation (when global step-size is small for convergence)
                    if i < self.n_outer:
                        o_i = order[i]
                        m_ray = int(m[o_i])
                        mean_ray, p_ray, w_ray, q_ray, s_ray = x[o_i], p[o_i], w[o_i], q[o_i][-m_ray:], s[o_i], 
                    elif i < self.n_outer*2: # to use recombination
                        o_i = order[i - self.n_outer]
                        mean_ray = (x[o_i] + xx)/2.0
                        p_ray = (p[o_i] + pp)/np.sqrt(2.0)
                        w_ray = (w[o_i] + ww)/np.sqrt(2.0)
                        m_ray = int(np.ceil((m[o_i] + mm)/2.0))
                        q_ray = (q[o_i][-m_ray:] + qq[-m_ray:])/np.sqrt(2.0)
                        s_ray = (s[o_i] + ss)/np.sqrt(2.0)
                    else:  # to mutate global step-size for diversity at meta-level
                        m_ray = int(self.rng_optimization.choice(self.m_list, p=self.p_list))
                        mean_ray, p_ray, w_ray, q_ray = xx, pp, ww, np.copy(qq[-m_ray:])
                        if i < self.n_outer*2 + (self.n_inner_es - self.n_outer*2)/5:
                            s_ray = self.rng_optimization.uniform(ss*0.3, ss*3.3)
                        else:
                            s_ray = self.rng_optimization.uniform(1e-16, 1e-15 + self.sigma)
                options[i] = {'mean': mean_ray, 'p': p_ray, 'w': w_ray, 'q': q_ray, 'sigma': s_ray,
                    'max_runtime': self.runtime_inner_es, 'fitness_threshold': self.fitness_threshold,
                    'seed_rng': self.rng_optimization.integers(0, np.iinfo(np.int64).max),
                    'verbose': False, 'saving_fitness': 100, 'm': m_ray}
                ray_es.append(ray_opt.remote(ray_problem, options[i]))
                ray_results.append(ray_es[i].optimize.remote(self.fitness_function, ray_args))
            results = ray.get(ray_results)  # to synchronize (a time-consuming operation)
            for i, r in enumerate(results):  # to run serially (clearly which should be light-weight)
                if self.best_so_far_y > r['best_so_far_y']:  # to update best-so-far solution and fitness
                    self.best_so_far_x, self.best_so_far_y = r['best_so_far_x'], r['best_so_far_y']
                x[i], y[i], p[i], w[i], q[i][-options[i]['m']:], s[i], m[i] =\
                    r['best_so_far_x'], r['best_so_far_y'], r['p'], r['w'], r['q'], r['sigma'], r['m']
                fit_start, fit_end = np.copy(r['fitness'][0]), np.copy(r['fitness'][-1])
                fit_start[0] += self.n_function_evaluations
                fit_end[0] += self.n_function_evaluations
                self.n_function_evaluations += r['n_function_evaluations']
                self.time_function_evaluations += r['time_function_evaluations']
                fitness.extend([fit_start, fit_end])
            order, is_first_generation = np.argsort(y)[:self.n_outer], False
            # to use *weighted multi-recombination* to update
            xx = np.dot(self._w_outer, x[order])  # for mean
            pp = np.dot(self._ww_outer, p[order])  # for evolution path
            ww = np.dot(self._ww_outer, w[order])  # for global step-size
            qq = np.zeros((q.shape[1], q.shape[2]))  # for covariance matrix
            for i in range(self.n_outer):
                qq[-int(m[order[i]]):] += self._ww_outer[i]*q[order[i]][-int(m[order[i]]):]
            ss = np.dot(self._ww_outer, s[order])  # for global step-size
            mm = np.dot(self._w_outer, m[order])
        ray.shutdown()  # to clear the current ray environment
        return self._collect(fitness)

    def _collect(self, fitness=None, y=None, mean=None):
        return {'best_so_far_x': self.best_so_far_x,
            'best_so_far_y': self.best_so_far_y,
            'n_function_evaluations': self.n_function_evaluations,
            'runtime': time.time() - self.start_time,
            'termination_signal': self.termination_signal,
            'time_function_evaluations': self.time_function_evaluations,
            'fitness': np.array(fitness)}
