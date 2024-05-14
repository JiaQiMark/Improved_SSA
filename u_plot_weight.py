import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体样式和大小
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 13

weight_min = 0
weight_max = 2
best_finess = -10
worst_fitness = 10

fitness = np.linspace(best_finess, worst_fitness, 200)

inertia_weight = (weight_min + (weight_max - weight_min) / 2
                  * (np.sin(np.pi * ((fitness - best_finess) / (worst_fitness - best_finess)) + np.pi / 2) + 1))
inverted_weight = (weight_min + weight_max) - inertia_weight

plt.plot(fitness, inertia_weight, label='Inertia Weight', linewidth=1.0, color='blue')
plt.plot(fitness, inverted_weight, label='Inverted Inertia Weight', linewidth=1.0, color='green', linestyle='--')
plt.xlabel("Fitness value")
plt.ylabel("Weight value")
plt.legend()
plt.show()
