import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
random.seed(0)

# from fitness_function import fitness_function


class GWO:
    def __init__(self, pack_size, iterations, range_l, range_u, vector_size, fitness_f):
        self.iterations = iterations
        self.pack_size = pack_size
        self.vector_size = vector_size
        self.min_range = range_l
        self.max_range = range_u
        self.fitness_function = fitness_f

    # generate wolf
    def wolf(self):
        wolf_position = [0.0 for i in range(self.vector_size)]
        for i in range(self.vector_size):
            wolf_position[i] = ((self.max_range - self.min_range) * random.random() + self.min_range)
        return wolf_position

    # generate wolf pack
    def pack(self):
        pack = [self.wolf() for i in range(self.pack_size)]
        return pack

    def hunt(self):
        # generate wolf pack
        wolf_pack = self.pack()
        # sort pack by fitness
        pack_fit = sorted([(self.fitness_function(i), i) for i in wolf_pack])
        convergence_fit = []
        # main loop
        for k in range(self.iterations):
            # choose best 3 solutions
            alpha, beta, delta = pack_fit[0][1], pack_fit[1][1], pack_fit[2][1]
            # print('GWO:', k+1, "/", self.iterations)
            convergence_fit.append(self.fitness_function(alpha))
            # linearly decreased from 2 to 0
            a = 2 * (1 - k / self.iterations)

            # updating each population member with the help of alpha, beta and delta
            for i in range(self.pack_size):
                # compute A and C
                A1, A2, A3 = a * (2 * random.random() - 1), a * (2 * random.random() - 1), a * (2 * random.random() - 1)
                C1, C2, C3 = 2 * random.random(), 2 * random.random(), 2 * random.random()

                # generate vectors for new position
                X1 = [0.0 for i in range(self.vector_size)]
                X2 = [0.0 for i in range(self.vector_size)]
                X3 = [0.0 for i in range(self.vector_size)]
                new_position = [0.0 for i in range(self.vector_size)]

                # hunting
                for j in range(self.vector_size):
                    X1[j] = alpha[j] - A1 * abs(C1 - alpha[j] - wolf_pack[i][j])
                    X2[j] = beta[j] - A2 * abs(C2 - beta[j] - wolf_pack[i][j])
                    X3[j] = delta[j] - A3 * abs(C3 - delta[j] - wolf_pack[i][j])
                    new_position[j] += X1[j] + X2[j] + X3[j]

                for j in range(self.vector_size):
                    new_position[j] /= 3.0
                    new_position[j] = np.clip(new_position[j], self.min_range, self.max_range)


                # fitness calculation of new position
                new_fitness = self.fitness_function(new_position)

                # if new position is better then replace, greedy update
                if new_fitness < self.fitness_function(wolf_pack[i]):
                    wolf_pack[i] = new_position

            # sort the new positions by their fitness
            pack_fit = sorted([(self.fitness_function(i), i) for i in wolf_pack])
        return convergence_fit


# from fitness_function import fitness_function, Dn, population_size, group_size, max_iterations, search_range


def gray_wolf_optimization(population_size, max_iterations, range_l, range_u, Dn, fitness_function):
    model = GWO(population_size, max_iterations, range_l, range_u, Dn, fitness_function)
    return model.hunt()



# convergence_fit = gray_wolf_optimization(population_size, max_iterations, search_range, Dn, fitness_function)
#
# iterations = np.linspace(0, max_iterations-1, len(convergence_fit), dtype=int)
# plt.yscale('log')
# plt.xlabel('iterations')
# plt.ylabel('fitness')
# plt.title('sparrow search algorithm')
# plt.plot(iterations, convergence_fit)
# plt.show()






