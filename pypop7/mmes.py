import numpy as np
from scipy.stats import norm

from pypoplib.es import ES


class MMES(ES):
    def __init__(self, problem, options):
        ES.__init__(self, problem, options)
        # set number of candidate direction vectors
        self.m = options.get('m')
        assert self.m > 0
        # set learning rate of evolution path
        self.c_c = options.get('c_c', 0.4/np.sqrt(self.ndim_problem))
        self.ms = options.get('ms', min(self.m, 4))  # mixing strength (l)
        assert self.ms > 0
        # set for paired test adaptation (PTA)
        self.c_s = options.get('c_s', 0.3)  # learning rate of global step-size adaptation
        self.a_z = options.get('a_z', 0.05)  # target significance level
        # set minimal distance of updating evolution paths (T)
        self.distance = options.get('distance', int(np.ceil(1.0/self.c_c)))
        # set success probability of geometric distribution (different from 4/n in the original paper)
        self.c_a = options.get('c_a', 3.8/self.ndim_problem)  # same as Matlab code
        self.gamma = options.get('gamma', 1.0 - np.power(1.0 - self.c_a, self.m))
        self._n_mirror_sampling = None
        self._z_1 = np.sqrt(1.0 - self.gamma)
        self._z_2 = np.sqrt(self.gamma/self.ms)
        self._p_1 = 1.0 - self.c_c
        self._p_2 = np.sqrt(self.c_c*(2.0 - self.c_c))
        self._w_1 = 1.0 - self.c_s
        self._w_2 = np.sqrt(self.c_s*(2.0 - self.c_s))

    def initialize(self, args=None, is_restart=False):
        self._n_mirror_sampling = int(np.ceil(self.n_individuals/2))
        x = np.zeros((self.n_individuals, self.ndim_problem))  # offspring population
        mean = np.copy(self.options.get('mean'))  # for parallelism
        p = np.copy(self.options.get('p'))  # for parallelism
        w = np.copy(self.options.get('w'))  # for parallelism
        q = np.copy(self.options.get('q'))  # for parallelism
        assert q.shape[0] == self.m
        t = np.zeros((self.m,))  # recorded generations
        v = np.arange(self.m)  # indexes to evolution paths
        y = np.tile(self._evaluate_fitness(x=mean, args=args), (self.n_individuals,))  # fitness
        return x, mean, p, w, q, t, v, y

    def iterate(self, x=None, mean=None, q=None, v=None, y=None, args=None):
        for k in range(self._n_mirror_sampling):  # mirror sampling
            zq = np.zeros((self.ndim_problem,))
            for _ in range(self.ms):
                j_k = v[(self.m - self.rng_optimization.geometric(self.c_a) % self.m) - 1]
                zq += self.rng_optimization.standard_normal()*q[j_k]
            z = self._z_1*self.rng_optimization.standard_normal((self.ndim_problem,))
            z += self._z_2*zq
            x[k] = mean + self.sigma*z
            if (self._n_mirror_sampling + k) < self.n_individuals:
                x[self._n_mirror_sampling + k] = mean - self.sigma*z
        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y
            y[k] = self._evaluate_fitness(x[k], args)
        return x, y

    def _update_distribution(self, x=None, mean=None, p=None, w=None, q=None,
                             t=None, v=None, y=None, y_bak=None):
        order = np.argsort(y)[:self.n_parents]
        y.sort()
        mean_w = np.dot(self._w[:self.n_parents], x[order])
        p = self._p_1*p + self._p_2*np.sqrt(self._mu_eff)*(mean_w - mean)/self.sigma
        mean = mean_w
        if self._n_generations < self.m:
            q[self._n_generations] = p
        else:
            k_star = np.argmin(t[v[1:]] - t[v[:(self.m - 1)]])
            k_star += 1
            if t[v[k_star]] - t[v[k_star - 1]] > self.distance:
                k_star = 0
            v = np.append(np.append(v[:k_star], v[(k_star + 1):]), v[k_star])
            t[v[-1]], q[v[-1]] = self._n_generations, p
        # conduct success-based mutation strength adaptation
        l_w = np.dot(self._w, y_bak[:self.n_parents] > y[:self.n_parents])
        w = self._w_1*w + self._w_2*np.sqrt(self._mu_eff)*(2*l_w - 1)
        self.sigma *= np.exp(norm.cdf(w) - 1.0 + self.a_z)
        return mean, p, w, q, t, v

    def restart_reinitialize(self, args=None, x=None, mean=None, p=None, w=None, q=None,
                             t=None, v=None, y=None, fitness=None):
        if self.is_restart and ES.restart_reinitialize(self, y):
            x, mean, p, w, q, t, v, y = self.initialize(args, True)
            self._print_verbose_info(fitness, y[0])
        return x, mean, p, w, q, t, v, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES.optimize(self, fitness_function)
        x, mean, p, w, q, t, v, y = self.initialize(args)
        self._print_verbose_info(fitness, y[0])
        while not self._check_terminations():
            y_bak = np.copy(y)
            # sample and evaluate offspring population
            x, y = self.iterate(x, mean, q, v, y, args)
            mean, p, w, q, t, v = self._update_distribution(x, mean, p, w, q, t, v, y, y_bak)
            self._n_generations += 1
            self._print_verbose_info(fitness, y)
            x, mean, p, w, q, t, v, y = self.restart_reinitialize(
                args, x, mean, p, w, q, t, v, y, fitness)
        results = self._collect(fitness, y, mean)
        results['p'] = p
        results['w'] = w
        results['q'] = q
        results['m'] = self.m
        return results
