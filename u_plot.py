import numpy as np
from tqdm import tqdm
import matplotlib


def draw(prefix_path, prefix_path_myssa, prefix_path_save,
         func_index,
         ssa, ssa_my, ssa_sin, ssa_pso, ssa_ic, ssa_wmr, gwo, pso, cso):
    import matplotlib.pyplot as plt
    # 设置全局字体样式和大小
    plt.figure(figsize=(6, 5))
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['font.size'] = 16

    epoch = 30
    max_iterations = 1000

    load_data_ssa     = np.zeros((epoch, max_iterations))
    load_data_myssa   = np.zeros((epoch, max_iterations))
    load_data_ssa_sin = np.zeros((epoch, max_iterations))
    load_data_ssa_pso = np.zeros((epoch, max_iterations))
    load_data_ssa_ic  = np.zeros((epoch, max_iterations))
    load_data_ssa_wmr = np.zeros((epoch, max_iterations))
    load_data_gwo     = np.zeros((epoch, max_iterations))
    load_data_pso     = np.zeros((epoch, max_iterations))
    load_data_cso     = np.zeros((epoch, max_iterations))

    if ssa:
        load_data_ssa     = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa.npy")
    if ssa_my:
        load_data_myssa   = np.load(f"{prefix_path_myssa}\\f{func_index}_{epoch}_{max_iterations}_myssa.npy")
    if ssa_sin:
        load_data_ssa_sin = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_sin.npy")
    if ssa_pso:
        load_data_ssa_pso = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_pso.npy")
    if ssa_ic:
        load_data_ssa_ic  = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_ic.npy")
    if ssa_wmr:
        load_data_ssa_wmr = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_wmr.npy")
    if gwo:
        load_data_gwo     = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_gwo.npy")
    if pso:
        load_data_pso     = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_pso.npy")
    if cso:
        load_data_cso     = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_cso.npy")

    curve_ssa     = np.zeros(max_iterations)
    curve_myssa   = np.zeros(max_iterations)
    curve_ssa_sin = np.zeros(max_iterations)
    curve_ssa_pso = np.zeros(max_iterations)
    curve_ssa_ic  = np.zeros(max_iterations)
    curve_ssa_wmr = np.zeros(max_iterations)
    curve_gwo     = np.zeros(max_iterations)
    curve_pso     = np.zeros(max_iterations)
    curve_cso     = np.zeros(max_iterations)

    if func_index in [17]:
        for i in range(epoch):
            curve_ssa     += load_data_ssa[i]
            curve_myssa   += load_data_myssa[i]
            curve_ssa_sin += load_data_ssa_sin[i]
            curve_ssa_pso += load_data_ssa_pso[i]
            curve_ssa_ic  += load_data_ssa_ic[i]
            curve_ssa_wmr += load_data_ssa_wmr[i]
            curve_gwo     += load_data_gwo[i]
            curve_pso     += load_data_pso[i]
            curve_cso     += load_data_cso[i]

        curve_ssa     /= len(load_data_ssa)
        curve_myssa   /= len(load_data_myssa)
        curve_ssa_sin /= len(load_data_ssa_sin)
        curve_ssa_pso /= len(load_data_ssa_pso)
        curve_ssa_ic  /= len(load_data_ssa_ic)
        curve_ssa_wmr /= len(load_data_ssa_wmr)
        curve_gwo     /= len(load_data_gwo)
        curve_pso     /= len(load_data_pso)
        curve_cso     /= len(load_data_cso)
    else:
        for i in range(len(load_data_ssa)):
            curve_ssa     += np.log10(load_data_ssa[i]     + 1e-320)
            curve_myssa   += np.log10(load_data_myssa[i]   + 1e-320)
            curve_ssa_sin += np.log10(load_data_ssa_sin[i] + 1e-320)
            curve_ssa_pso += np.log10(load_data_ssa_pso[i] + 1e-320)
            curve_ssa_ic  += np.log10(load_data_ssa_ic[i]  + 1e-320)
            curve_ssa_wmr += np.log10(load_data_ssa_wmr[i] + 1e-320)
            curve_gwo     += np.log10(load_data_gwo[i]     + 1e-320)
            curve_pso     += np.log10(load_data_pso[i]     + 1e-320)
            curve_cso     += np.log10(load_data_cso[i]     + 1e-320)

        curve_ssa     /= len(load_data_ssa)
        curve_myssa   /= len(load_data_myssa)
        curve_ssa_sin /= len(load_data_ssa_sin)
        curve_ssa_pso /= len(load_data_ssa_pso)
        curve_ssa_ic  /= len(load_data_ssa_ic)
        curve_ssa_wmr /= len(load_data_ssa_wmr)
        curve_gwo     /= len(load_data_gwo)
        curve_pso     /= len(load_data_pso)
        curve_cso     /= len(load_data_cso)

        curve_ssa     = 10 ** curve_ssa
        curve_myssa   = 10 ** curve_myssa
        curve_ssa_sin = 10 ** curve_ssa_sin
        curve_ssa_pso = 10 ** curve_ssa_pso
        curve_ssa_ic  = 10 ** curve_ssa_ic
        curve_ssa_wmr = 10 ** curve_ssa_wmr
        curve_gwo     = 10 ** curve_gwo
        curve_pso     = 10 ** curve_pso
        curve_cso     = 10 ** curve_cso
        plt.yscale('log')

    plt.xlabel('Iteration number')
    plt.ylabel('Fitness value')
    x = range(max_iterations)
    if ssa_ic:
        plt.plot(curve_ssa_ic, label='SSA-W1', linewidth=0.5, color='cyan', linestyle='--')
    if ssa_wmr:
        plt.plot(curve_ssa_wmr, label='SSA-W2', linewidth=0.5, color='black', linestyle='--')

    if ssa:
        plt.plot(curve_ssa, label='SSA', linewidth=0.5, color='blue', linestyle='--')
    if gwo:
        plt.plot(curve_gwo, label='GWO', linewidth=0.5, color='red', linestyle='--')
    if pso:
        plt.plot(curve_pso, label='PSO', linewidth=0.5, color='y', linestyle='--')
    if cso:
        plt.plot(curve_cso, label='CSO', linewidth=0.5, color='black', linestyle='--')

    if ssa_sin:
        plt.plot(curve_ssa_sin, label='SCA-CSSA', linewidth=0.5, color='green', linestyle='--')
    if ssa_pso:
        plt.plot(curve_ssa_pso, label='HSSA', linewidth=0.5, color='red', linestyle='--')
    if ssa_my:
        plt.plot(curve_myssa, label='CIW-SSA', linewidth=0.5, color='b', linestyle='--', marker='^', markevery=50)

    plt.legend()
    if func_index in [15, 16, 17]:
        plt.title(f"f {func_index - 1}")
    elif func_index in [19, 20]:
        plt.title(f"f {func_index - 2}")
    else:
        plt.title(f"f {func_index}")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{prefix_path_save}//f{func_index}_{epoch}_{max_iterations}.png',
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.clf()


pref_path = "result_3_low_dimension_30"
pref_path_myssa = "result_6_ld30_myssa"
# pref_path_myssa = "result_2_only_weight_strategy"
# pref_path_save = "picture_4_compare_improved"
# pref_path_save = "picture_5_only18_function"
pref_path_save = "picture_5_only18_compare_improved"


function_index_list = [0,
                       1,   2,  3,  4,  5,  6,  7,  8,  9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

for i in range(1, len(function_index_list)):
    if i in [14, 18]:
        continue
    function_index = function_index_list[i]
    f_ssa_ic = 0
    f_ssa_wmr = 0

    f_ssa = 0
    f_gwo = 0
    f_pso = 0
    f_cso = 0

    f_ssa_pso = 1
    f_ssa_sin = 1

    f_ciw_ssa = 1
    draw(pref_path, pref_path_myssa, pref_path_save,
         function_index,
         f_ssa, f_ciw_ssa, f_ssa_sin, f_ssa_pso, f_ssa_ic, f_ssa_wmr, f_gwo, f_pso, f_cso)




