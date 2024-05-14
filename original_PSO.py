import numpy as np
import matplotlib.pyplot as plt


class PSO:
    def __init__(self, func, n_dim=30, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.8, c1=1.5, c2=1.5):

        self.func = func
        self.w = w                 # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.pop = pop             # number of particles
        self.n_dim = n_dim         # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter   # max iter

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        v_high = self.ub - self.lb

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.cal_y()                      # y = f(x) for all particles
        self.pbest_x = self.X.copy()               # personal best location of every particle in history
        self.pbest_y = self.Y.copy()               # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0)   # global best location for all particles
        self.gbest_y = np.inf                      # global best y for all particles
        self.gbest_y_list = []                     # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        self.need_update = None


    def update_v(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = (self.w * self.V +
                  self.cp * r1 * (self.pbest_x - self.X) +
                  self.cg * r2 * (self.gbest_x - self.X))

    def update_x(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        temp_y = np.zeros(self.pop)
        for i in range(self.pop):
            temp_y[i] = self.func(self.X[i])

        self.Y = temp_y.reshape(-1, 1)

    def update_pbest(self):
        self.need_update = self.pbest_y > self.Y

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        idx_min = self.pbest_y.argmin()
        if self.pbest_y[idx_min] < self.gbest_y:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None, precision=None, N=20):
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.update_v()
            self.recorder()
            self.update_x()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0

            # print('Iter: {}, Best fit: {}'.format(iter_num, self.gbest_y))
            # self.gbest_y_list.append(self.gbest_y)
            self.gbest_y_list.append(np.min(self.pbest_y))
            # print('PSO:', iter_num + 1, "/", self.max_iter)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y


# from fitness_function import fitness_function, Dn, population_size, group_size, max_iterations, search_range
def particle_swarm_optimization(pop_size, max_iter, range_l, range_u, D, func):
    pso = PSO(func, D, pop_size, max_iter, lb=range_l, ub=range_u, w=0.8, c1=1.5, c2=1.5)
    pso.run()
    return pso.gbest_y_list



# convergence_fit = particle_swarm_optimization(fitness_function, Dn, population_size, max_iterations, search_range)
#
# plt.plot(convergence_fit)
# plt.yscale("log")
# plt.show()



