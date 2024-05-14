import matplotlib.pyplot as plt
import numpy as np
from fitness_function import select_fitness_function, search_range_list, optimal_list


func_index = 20
search_range = search_range_list[func_index]


def fitness_function(x):
    return select_fitness_function(func_index, x)


def paint_image_2d():
    num_sample = 100
    test_x = np.linspace(search_range[0], search_range[1], num_sample, dtype=float)
    test_x = test_x.reshape((-1, 1))
    test_y = []

    for i in range(num_sample):
        test_y.append(fitness_function(test_x[i]))
    # 设置图像标题
    plt.title('三维曲面图')
    plt.plot(test_x, test_y)
    # 显示图像
    plt.show()
    print(test_y)


def paint_image_3d():
    num_sample = 200
    test_x = np.linspace(search_range[0], search_range[1], num_sample, dtype=float)
    test_y = np.linspace(search_range[0], search_range[1], num_sample, dtype=float)
    test_x, test_y = np.meshgrid(test_x, test_y)

    test_z = np.ndarray(shape=(num_sample, num_sample), dtype=float)
    for i in range(num_sample):
        for j in range(num_sample):
            test_z[i][j] = fitness_function(np.array([test_x[i][j], test_y[i][j]]))

    # 创建一个三维坐标轴对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 使用add_subplot创建3D坐标轴

    # 绘制三维曲面图
    surf = ax.plot_surface(test_x, test_y, test_z, cmap='viridis')

    # 添加颜色条
    fig.colorbar(surf)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 设置图像标题

    plt.title(f'f{func_index}  optimal:{optimal_list[func_index]:.4f}')
    # 显示图像
    optimal = np.min(np.array(test_z))
    print(optimal)
    plt.show()


paint_image_3d()



