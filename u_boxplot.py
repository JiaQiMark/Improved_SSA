import numpy as np
from docx import Document
import matplotlib


def draw_box(prefix_path, prefix_path_myssa, prefix_path_save,
             func_index,
             ssa, ssa_my, ssa_sin, ssa_pso, ssa_ic, ssa_wmr, gwo, pso, cso):
    import matplotlib.pyplot as plt
    epoch = 30
    max_iterations = 1000
    # 设置全局字体样式和大小
    plt.figure(figsize=(6, 5))
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['font.size'] = 16

    box_list = []

    if ssa_ic:
        load_data_ssa_ic  = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_ic.npy")
        last_row = load_data_ssa_ic[:, -1]

    if ssa_wmr:
        load_data_ssa_wmr = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_wmr.npy")
        last_row = load_data_ssa_wmr[:, -1]

    if ssa:
        load_data_ssa     = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa.npy")
        last_row = load_data_ssa[:, -1]

    if gwo:
        load_data_gwo     = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_gwo.npy")
        last_row = load_data_gwo[:, -1]

    if pso:
        load_data_pso     = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_pso.npy")
        last_row = load_data_pso[:, -1]

    if cso:
        load_data_cso     = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_cso.npy")

    if ssa_sin:
        load_data_ssa_sin = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_sin.npy")
        last_row_ssa_sin = load_data_ssa_sin[:, -1]
        box_list.append(last_row_ssa_sin)
    if ssa_pso:
        load_data_ssa_pso = np.load(f"{prefix_path}\\f{func_index}_{epoch}_{max_iterations}_ssa_pso.npy")
        last_row_ssa_pso = load_data_ssa_pso[:, -1]
        box_list.append(last_row_ssa_pso)
    if ssa_my:
        load_data_myssa   = np.load(f"{prefix_path_myssa}\\f{func_index}_{epoch}_{max_iterations}_myssa.npy")
        last_row_ssa_my = load_data_myssa[:, -1]
        box_list.append(last_row_ssa_my)

    # if func_index != 17:
        # plt.yscale('log')
    # 创建一个箱形图
    plt.boxplot([box_list[0], box_list[1], box_list[2]])
    # 设置箱形图的标签
    plt.xticks([1, 2, 3], ['HSSA', 'SCA-CSSA', 'CIW-SSA'])
    plt.ylabel('Fitness value')

    if func_index in [15, 16, 17]:
        plt.title(f"f {func_index - 1}")
    elif func_index in [19, 20]:
        plt.title(f"f {func_index - 2}")
    else:
        plt.title(f"f {func_index}")

    plt.tight_layout()
    plt.show()


pref_path = "result_3_low_dimension_30"
pref_path_myssa = "result_6_ld30_myssa"
pref_path_save = "D:\\001__studyMetarial\\05__mypaper\\best_mean_std\\02_SCA-CSSA.docx"


function_index_list = [0,
                       1,   2,  3,  4,  5,  6,  7,  8,  9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# for i in [9, 11, 16, 17]:
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

    f_ssa_sin = 1
    f_ssa_pso = 1
    f_ssa_my = 1

    draw_box(pref_path, pref_path_myssa, pref_path_save, function_index,
             f_ssa, f_ssa_my, f_ssa_sin, f_ssa_pso, f_ssa_ic, f_ssa_wmr, f_gwo, f_pso, f_cso)




