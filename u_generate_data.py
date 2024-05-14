import numpy as np
from tqdm import tqdm
import time
from original_GWO import gray_wolf_optimization
from original_PSO import particle_swarm_optimization
from original_CSO import implementing_cso
from SSA import sparrow_search_optimization
from SSA_my_temp import my_sparrow_search_optimization
from SSA_sin import sin_sparrow_search_optimization
from SSA_pso import pso_sparrow_search_optimization
from SSA_ic import icssa_sparrow_search_optimization
from SSA_wmr import wmr_sparrow_search_optimization

from fitness_function import select_fitness_function, search_range_list, dimension_list


def calculate(prefix_path, prefix_path_myssa,
              func_index, ssa, ssa_my, ssa_sin, ssa_pso, ssa_ic, ssa_wmr, gwo, pso, cso):
    Dn = dimension_list[func_index]
    search_range = search_range_list[func_index]
    max_iterations = 1000
    population_size = 100
    epoch = 30

    def fitness_function(x):
        return select_fitness_function(func_index, x)

    ultimate_result_ssa     = np.zeros((epoch, max_iterations))
    ultimate_result_myssa   = np.zeros((epoch, max_iterations))
    ultimate_result_ssa_sin = np.zeros((epoch, max_iterations))
    ultimate_result_ssa_pso = np.zeros((epoch, max_iterations))
    ultimate_result_ssa_ic  = np.zeros((epoch, max_iterations))
    ultimate_result_ssa_wmr = np.zeros((epoch, max_iterations))
    ultimate_result_gwo     = np.zeros((epoch, max_iterations))
    ultimate_result_pso     = np.zeros((epoch, max_iterations))
    ultimate_result_cso     = np.zeros((epoch, max_iterations))

    start_time = time.time()

    if ssa:
        for e in tqdm(range(epoch), desc=f"Processing_function_{func_index}_ssa"):
            temp_ndarray = sparrow_search_optimization(population_size,
                                                       max_iterations,
                                                       search_range[0],
                                                       search_range[1],
                                                       Dn,
                                                       fitness_function, 0)
            ultimate_result_ssa[e] = temp_ndarray
        np.save(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa.npy", ultimate_result_ssa)

    if ssa_my:
        for e in tqdm(range(epoch), desc=f"Processing_function_{func_index}_ssa_my"):
            temp_ndarray = my_sparrow_search_optimization(population_size,
                                                          max_iterations,
                                                          search_range[0],
                                                          search_range[1],
                                                          Dn,
                                                          fitness_function, 0)
            ultimate_result_myssa[e] = temp_ndarray
        np.save(f"{prefix_path_myssa}\\f{func_index}_{epoch}_{max_iterations}_myssa.npy", ultimate_result_myssa)

    if ssa_sin:
        for e in tqdm(range(epoch), desc=f"Processing_function_{func_index}_ssa_sin"):
            temp_ndarray = sin_sparrow_search_optimization(population_size,
                                                           max_iterations,
                                                           search_range[0],
                                                           search_range[1],
                                                           Dn,
                                                           fitness_function, 0)
            ultimate_result_ssa_sin[e] = temp_ndarray
        np.save(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_sin.npy", ultimate_result_ssa_sin)

    if ssa_pso:
        for e in tqdm(range(epoch), desc=f"Processing_function_{func_index}_ssa_pso"):
            temp_ndarray = pso_sparrow_search_optimization(population_size,
                                                           max_iterations,
                                                           search_range[0],
                                                           search_range[1],
                                                           Dn,
                                                           fitness_function, 0)
            ultimate_result_ssa_pso[e] = temp_ndarray
        np.save(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_pso.npy", ultimate_result_ssa_pso)

    if ssa_ic:
        for e in tqdm(range(epoch), desc=f"Processing_function_{func_index}_ssa_ic"):
            temp_ndarray = icssa_sparrow_search_optimization(population_size,
                                                             max_iterations,
                                                             search_range[0],
                                                             search_range[1],
                                                             Dn,
                                                             fitness_function, 0)
            ultimate_result_ssa_ic[e] = temp_ndarray
        np.save(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_ic.npy", ultimate_result_ssa_ic)

    if ssa_wmr:
        for e in tqdm(range(epoch), desc=f"Processing_function_{func_index}_ssa_wmr"):
            temp_ndarray = wmr_sparrow_search_optimization(population_size,
                                                           max_iterations,
                                                           search_range[0],
                                                           search_range[1],
                                                           Dn,
                                                           fitness_function, 0)
            ultimate_result_ssa_wmr[e] = temp_ndarray
        np.save(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_wmr.npy", ultimate_result_ssa_wmr)

    if gwo:
        for e in tqdm(range(epoch), desc=f"Processing_function_{func_index}_gwo"):
            temp_ndarray = gray_wolf_optimization(population_size,
                                                  max_iterations,
                                                  search_range[0],
                                                  search_range[1],
                                                  Dn,
                                                  fitness_function)
            ultimate_result_gwo[e] = temp_ndarray
        np.save(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_gwo.npy", ultimate_result_gwo)

    if pso:
        for e in tqdm(range(epoch), desc=f"Processing_function_{func_index}_pso"):
            temp_ndarray = particle_swarm_optimization(population_size,
                                                       max_iterations,
                                                       search_range[0],
                                                       search_range[1],
                                                       Dn,
                                                       fitness_function)
            ultimate_result_pso[e] = temp_ndarray
        np.save(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_pso.npy", ultimate_result_pso)

    if cso:
        for e in tqdm(range(epoch), desc=f"Processing_function_{func_index}_cso"):
            temp_ndarray = implementing_cso(population_size,
                                            max_iterations,
                                            search_range[0],
                                            search_range[1],
                                            Dn,
                                            fitness_function,
                                            10,
                                            40)
            ultimate_result_cso[e] = temp_ndarray
        np.save(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_cso.npy", ultimate_result_cso)

    end_time = time.time()
    second_ms = int(end_time - start_time)
    print(f"{second_ms//60}m {second_ms%60}s")



prefix_path = "result_3_low_dimension_30"
prefix_path_myssa = "result_8_hd500_myssa"

function_index_list = [0,
                       1,   2,  3,  4,  5,  6,  7,  8,  9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


for i in range(11, 16):
# for i in [1]:
    index = function_index_list[i]
    ssa = 0
    gwo = 0
    pso = 0
    cso = 0
    ssa_my = 1
    ssa_sin = 0
    ssa_pso = 0
    ssa_ic = 0
    ssa_wmr = 0
    calculate(prefix_path, prefix_path_myssa,
              index, ssa, ssa_my, ssa_sin, ssa_pso, ssa_ic, ssa_wmr, gwo, pso, cso)
print("over!!!!!!!!!!!!!!!!!!")


