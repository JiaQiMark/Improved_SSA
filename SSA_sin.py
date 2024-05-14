import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from fitness_function import select_fitness_function

func_index = 1
search_range_list = [0,
                     100, 1, 10, 0, 0,
                     0,   0,  0, 0, 0,
                     0,   0, 600, 5.12, 500]
search_range = search_range_list[func_index]
max_iterations = 500
Dn = 30
population_size = 90
group_size = 9


def fitness_function(x):
    return select_fitness_function(func_index, x)


def sin_sparrow_search_optimization(population, max_iterations, search_num_l, search_num_u, dim, fitness_function, flag_p):
    ST = 0.8

    propotion_alerter = 0.2
    # The propotion of producer
    propotion_producer = 0.2
    producer_num = round(population * propotion_producer)
    low_bundary = search_num_l * np.ones((1, dim))
    up_bundary = search_num_u * np.ones((1, dim))

    # 代表麻雀位置
    position = np.zeros((population, dim))
    x1 = np.zeros((population, dim))
    x2 = np.zeros((population, dim))
    x1_f = np.zeros(population)
    x2_f = np.zeros(population)

    # 适应度初始化
    fitness = np.zeros(population)

    for i in range(population):
        position[i, :] = low_bundary + (up_bundary - low_bundary) * np.random.rand(1, dim)
        fitness[i] = fitness_function(position[i, :])

    # 初始化收敛曲线
    convergence_curve = np.zeros(max_iterations)

    # for t in tqdm(range(max_iterations), desc="SCA-CSSA", miniters=max_iterations/5):
    for t in range(max_iterations):
        # 对麻雀的适应度值进行排序，并取出下标
        fitness_sorted_index = np.argsort(fitness.T)
        best_finess = np.min(fitness)
        best_finess_index = np.argmin(fitness)
        best_position = position[best_finess_index, :]

        worst_fitness = np.max(fitness)
        worst_fitness_index = np.argmax(fitness)
        worst_positon = position[worst_fitness_index, :]

        # 1) 全部个体位置更新
        R2 = np.random.rand(1)
        for i in range(population):
            p_i = fitness_sorted_index[i]
            if R2 < ST:
                alaph = np.random.rand()
                x1[p_i, :] = position[p_i, :] * np.exp(-i / (alaph * max_iterations))
            elif R2 >= ST:
                q = np.random.normal(0, 1, 1)
                l_dim = np.ones((1, dim))
                x1[p_i, :] = position[p_i, :] + q * l_dim

            # 越界处理
            x1[p_i, :] = np.clip(position[p_i, :], search_num_l, search_num_u)
            x1_f[p_i] = fitness_function(x1[p_i, :])

        # 找出最优的”探索者“
        next_best_position_index = np.argmin(fitness[:])
        next_best_position = position[next_best_position_index, :]

        # 2) 全部个体位置更新
        for i in range(0, population):
            s_i = fitness_sorted_index[i]
            if s_i > (population / 2):
                q = np.random.normal(0, 1, 1)
                x2[s_i, :] = q * np.exp((worst_positon - position[s_i, :])/(s_i**2))
            else:
                l_dim = np.ones((1, dim))
                a = np.floor(np.random.rand(1, dim) * 2) * 2 - 1
                a_plus = 1 / (a.T * np.dot(a, a.T))
                x2[s_i, :] = next_best_position + l_dim * np.dot(np.abs(position[s_i, :] - next_best_position),
                                                                       a_plus)
            # # 越界处理
            # x2[s_i, :] = np.clip(position[s_i, :], search_num_l, search_num_u)
            # x2_f[s_i] = fitness_function(x2[s_i, :])

        # 全部个体位置更新
        for i in range(population):
            w1, w2 = 0, 0
            if x1_f[i] == best_finess:
                w1 = 2
            elif x2_f[i] == best_finess:
                w2 = 2
            else:
                w1 = 1
                w2 = 1

            x_cooperate = (w1 * x1[i] + w2 * x2[i]) / 2
            x_cooperate = np.clip(x_cooperate, search_num_l, search_num_u)
            new_fitness = fitness_function(x_cooperate)
            if new_fitness < fitness[i]:
                position[i] = x_cooperate
                fitness[i] = new_fitness

        # 3) 意识到危险的麻雀的位置更新
        arrc = np.arange(len(fitness_sorted_index[:]))
        # 随机排列序列
        random_arrc = np.random.permutation(arrc)
        # 随机选取警戒者
        num_alerter = round(propotion_alerter * population)
        alerter_index = fitness_sorted_index[random_arrc[0:num_alerter]]

        for i in range(num_alerter):
            a_i = alerter_index[i]
            f_i = fitness[a_i]
            f_g = best_finess
            f_w = worst_fitness
            if f_i > f_g:
                beta = np.random.normal(0, 1 , 1)
                position[a_i, :] = best_position + beta * np.abs(position[a_i, :] - best_position)
            elif f_i == f_g:
                e = 1e-20
                k = np.random.uniform(-1, 1, 1)
                position[a_i, :] = position[a_i, :] + k * ((np.abs(position[a_i, :] - worst_positon)) /
                                                           (f_i - f_w + e))
            # 越界处理
            position[a_i, :] = np.clip(position[a_i, :], search_num_l, search_num_u)
            fitness[a_i] = fitness_function(position[a_i, :])

        # sin cos process
        for i in range(population):
            a = 2
            r1 = a - t * (a / max_iterations)
            r2 = np.random.uniform(0, 2 * math.pi)
            r3 = np.random.uniform(0, 2)
            r4 = np.random.rand()
            sequence = [abs(x) for x in (r3 * best_position - position[i])]
            if r4 < 0.5:
                x_ssa = position[i] + [r1 * math.sin(r2) * item for item in sequence]
            else:
                x_ssa = position[i] + [r1 * math.cos(r2) * item for item in sequence]

            x_ssa = np.clip(x_ssa, search_num_l, search_num_u)
            new_fitness = fitness_function(x_ssa)
            if new_fitness < fitness[i]:
                position[i] = x_ssa
                fitness[i] = new_fitness

        # 一次比一次好机制
        if t == 0:
            convergence_curve[t] = np.min(fitness)
        else:
            convergence_curve[t] = min(np.min(fitness), convergence_curve[t-1])
        if flag_p == 1:
            print("SSA_sincos", t + 1, " / ", max_iterations)
    return convergence_curve


# sin_convergence_fit = sin_sparrow_search_optimization(population_size,
#                                                       max_iterations,
#                                                       -search_range,
#                                                       search_range,
#                                                       Dn,
#                                                       fitness_function)
#
# iterations = np.linspace(0, max_iterations-1, len(sin_convergence_fit), dtype=int)
# plt.yscale('log')
# plt.xlabel('iterations')
# plt.ylabel('fitness')
# plt.title('sparrow search algorithm')
# plt.plot(iterations, sin_convergence_fit)
# plt.show()




