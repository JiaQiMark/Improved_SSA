import numpy as np
import matplotlib.pyplot as plt
from fitness_function import select_fitness_function, search_range_list, dimension_list
from SSA import sparrow_search_optimization
from SSA_ic import icssa_sparrow_search_optimization
from SSA_wmr import wmr_sparrow_search_optimization
from SSA_my import my_sparrow_search_optimization


func_index = 1
Dn = dimension_list[func_index]
search_range = search_range_list[func_index]
max_iterations = 500
population_size = 100


def fitness_function(x):
    return select_fitness_function(func_index, x)


fitness_ssa = sparrow_search_optimization(population_size,
                                          max_iterations,
                                          search_range[0],
                                          search_range[1],
                                          Dn,
                                          fitness_function, 1)


fitness_ssamy = my_sparrow_search_optimization(population_size,
                                               max_iterations,
                                               search_range[0],
                                               search_range[1],
                                               Dn,
                                               fitness_function, 1)

fitness_ssa_ic = icssa_sparrow_search_optimization(population_size,
                                                   max_iterations,
                                                   search_range[0],
                                                   search_range[1],
                                                   Dn,
                                                   fitness_function, 1)

fitness_ssa_wmr = wmr_sparrow_search_optimization(population_size,
                                                  max_iterations,
                                                  search_range[0],
                                                  search_range[1],
                                                  Dn,
                                                  fitness_function, 1)

plt.plot(fitness_ssa, label='SSA', linewidth=0.5, color='blue', linestyle='--')
plt.plot(fitness_ssa_ic, label='ICSSA', linewidth=0.5, color='red', linestyle='--')
plt.plot(fitness_ssa_wmr, label='WMR-SSA', linewidth=0.5, color='green', linestyle='--')
plt.plot(fitness_ssamy, label='SSA_MY', linewidth=0.5, color='blue', linestyle='--', marker='^')

plt.legend()
plt.title(f"f{func_index}")
plt.yscale("log")
plt.show()

