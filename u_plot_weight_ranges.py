import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt


def draw(prefix_path_myssa, prefix_path_save, func_index):
    # 设置全局字体样式和大小
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['font.size'] = 15
    epoch = 10
    max_iterations = 1000

    w_range_0 = f"W_min_0__max_2"
    w_range_1 = f"W_min_0.1__max_1.9"
    w_range_2 = f"W_min_0.2__max_1.8"
    w_range_3 = f"W_min_0.3__max_1.7"
    w_range_4 = f"W_min_0.4__max_1.6"
    w_range_5 = f"W_min_0.5__max_1.5"
    w_range_6 = f"W_min_0.6__max_1.4"
    w_range_7 = f"W_min_0.7__max_1.3"
    w_range_8 = f"W_min_0.8__max_1.2"
    w_range_9 = f"W_min_0.9__max_1.1"

    range_list = [w_range_0, w_range_1, w_range_2, w_range_3, w_range_4,
                  w_range_5, w_range_6, w_range_7, w_range_8, w_range_9]
    last_optimal = []
    for idx in range(0, 10):
        suff_range = range_list[idx]
        load_data = np.load(f"{prefix_path_myssa}\\f{func_index}_{epoch}_{max_iterations}_myssa_{suff_range}.npy")

        curve = np.zeros(len(load_data[0]))
        for tr in range(len(load_data)):
            curve += np.log10(load_data[tr] + 1e-320)

        curve /= len(load_data)
        curve_myssa = 10 ** curve

        last_optimal.append("%.4E" % curve_myssa[-1])

        if idx == 0:
            plt.plot(curve_myssa, label='weight_min=0, weight_max=2', linewidth=0.5, color='b', linestyle='--', marker='^', markevery=50)
        elif idx == 1:
            plt.plot(curve_myssa, label='others', linewidth=0.5, color='g', linestyle='--', markevery=50)
        else:
            plt.plot(curve_myssa, label='', linewidth=0.5, color='g', linestyle='--', markevery=50)

    print(last_optimal)

    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness value')
    plt.legend()
    plt.title(f"f{func_index}")
    plt.show()
    # plt.savefig(f'{prefix_path_save}//f{func_index}_{epoch}_{max_iterations}.png')
    # plt.clf()




prefix_path_myssa = "result_1_select_W_range"
prefix_path_save = "picture"


function_index_list = [0,
                       1,   2,  3,  4,  5,  6,  7,  8,  9, 10,
                       11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# [1, 2, 12, 16]
for i in tqdm([1], "ploting..."):
    function_index = function_index_list[i]
    draw(prefix_path_myssa, prefix_path_save, function_index)




