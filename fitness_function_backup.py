import numpy as np
import math

search_range_list = [[0, 0],
                     [-100, 100],       [-1, 1],       [-10, 10],       [-10, 10],     [-100, 100],
                     [-100, 100],     [-30, 30],     [-600, 600],       [-10, 10],   [-1.28, 1.28],
                     [-100, 100],   [-100, 100], [-2.048, 2.048],   [-5.12, 5.12],     [-500, 500],
                     [-32, 32],     [-600, 600],       [-10, 10],       [-50, 50],         [-1, 1],
                     [-5, 5],           [0, 14],       [-10, 10],       [-20, 20],     [-4.5, 4.5],
                     [-10, 10],     [-512, 512],       [-10, 10],       [-10, 10],       [-10, 10]]


common_d = 30
dimension_list = [0,
                  common_d, common_d, common_d, common_d, common_d,
                  common_d, common_d, common_d, common_d, common_d,
                  common_d, common_d, common_d, common_d, common_d,
                  common_d, common_d, common_d, common_d, common_d,
                  2,                2,       2,        2,        2,
                  4,                2,       2,        2,        2]

optimal_list = [0,
                0,        0,         0,         0,         0,
                0,        0,         0,         0,         0,
                0,        0,         0,         0,    -12540,
                0,        0,         0,         0,         0,
                -1.0316,  0,      -1.0,      -1.0,         0,
                -10.5364, 0,         0,         0,         0]



def fitness_function_1(x):
    dim = len(x)
    y = 0
    for i in range(0, dim):
        y += x[i]**2
    return y


def fitness_function_2(x):
    y = 0
    for i in range(len(x)):
        y += np.abs(x[i])**(i + 2)
    return y


def fitness_function_3(x):
    y = 0
    for i in range(len(x)):
        y += (i + 1) * (x[i]**2)
    return y


def fitness_function_4(x):
    y = 0
    for i in range(len(x)):
        y += abs(x[i])

    m = 1
    for i in range(len(x)):
        m *= abs(x[i])
    return y + m


def fitness_function_5(x):
    y = 0
    for i in range(len(x)):
        y2 = 0
        for j in range(i):
            y2 += x[i]
        y += y2**2
    return y


def fitness_function_6(x):
    abs_x = np.abs(x)
    return np.max(abs_x)


def fitness_function_7(x):
    summary = 0
    for i in range(len(x) - 1):
        summary += (100 * (x[i+1] - x[i]) ** 2 + (x[i] - 1) ** 2)

    return summary


def fitness_function_8(x):
    dim = len(x)
    conti_addition = 0
    for i in range(1, dim+1):
        conti_addition += x[i-1]**2

    conti_multiple = 1
    for i in range(1, dim+1):
        conti_multiple *= math.cos(x[i-1] / math.sqrt(i))
    return conti_addition / 4000 - conti_multiple + 1


def fitness_function_9(x):
    sum1 = 0
    sum2 = 0

    for i in range(len(x)):
        sum1 += (x[i] ** 2)
        sum2 += (0.5 * (i + 1) * x[i])

    result = sum1 + sum2**2  + sum2**4
    return result


def fitness_function_10(x):
    sum1 = 0
    for i in range(len(x)):
        sum1 += (i+1) * x[i]**4

    return sum1 * np.random.rand()


def fitness_function_11(x):
    sum1 = 0
    for i in range(len(x)):
        sum1 += abs(x[i] + 0.5) ** 2
    return sum1


def fitness_function_12(x):
    sum1 = 0
    for i in range(len(x)):
        sum1 += x[i]**2
    return (10**6) * (x[0]**2) * sum1









def fitness_function_13(x):
    sum1 = 0
    for i in range(len(x) - 1):
        sum1 += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return sum1


def fitness_function_14(x):
    dim = len(x)
    continuous_addition = 0
    for i in range(dim):
        continuous_addition += (x[i]**2 - 10 * math.cos(2 * math.pi * x[i]) + 10)
    return continuous_addition


def fitness_function_15(x):
    dim = len(x)
    continuous_addition = 0
    for i in range(dim):
        continuous_addition += -1 * x[i] * math.sin(math.sqrt(abs(x[i])))
    return continuous_addition


def fitness_function_16(x):
    sum1 = 0
    sum2 = 0
    for i in range(len(x)):
        sum1 += x[i]**2
        sum2 += math.cos(2 * math.pi * x[i])

    result = -20 * np.exp(-0.2 * np.sqrt(sum1 / len(x))) + 20 + math.e - np.exp(sum2 / len(x))
    return result


def fitness_function_17(x):
    summary = 0
    product = 1
    for i in range(len(x)):
        summary += x[i]**2
        product *= math.cos(x[i] / math.sqrt(i+1))

    result = summary / 4000 - product + 1
    return result


def fitness_function_18(x):
    summary = 0
    for i in range(len(x)):
        summary += abs(x[i] * math.sin(x[i]) + 0.1 * x[i])

    return summary


def fitness_function_19(x):
    sum1 = 0
    for i in range(len(x) - 1):
        yi = 1 + (x[i] + 1)/4
        yn = 1 + (x[i+1] + 1)/4
        sum1 += (yi - 1)**2 * (1 + 10 * (math.sin(math.pi * yn)**2))

    sum2 = 0
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = 1 + (x[i] + 1)/4
        a, k, m, u = 10, 100, 4, 0
        if x[i] > a:
            u = k * (x[i] - a)**m
        elif (x[i] > -a) and (x[i] < a):
            u = 0
        elif x[i] < -a:
            u = k * (-x[i] - a)**m
        sum2 += u

    n = len(x)
    return (math.pi / n) * (10 * math.sin(math.pi * y[0]) + sum1 + (y[n-1] - 1)**2) + sum2


def fitness_function_20(x):
    summary = 0
    for i in range(len(x)):
        a, b = 0, 0
        try:
            a = x[i]**6
            b = 2 + np.sin(1 / (x[i]))
        except ZeroDivisionError:
            a = x[i] ** 6
            b = 2
            print(f"MJQ_ZeroDivisionError__x:{x[i]}")
        except ValueError:
            a = x[i] ** 6
            b = 2
            print(f"MJQ_ValueError__x:{x[i]}")
        else:
            a = x[i] ** 6
            b = 2 + np.sin(1 / (x[i]))
        finally:
            summary += a * b
    return summary














def fitness_function_21(x):
    summary = 4 * x[0]**2 + 2.1 * x[0]**4 + (x[0]**6)/3 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1]**4
    return summary


def fitness_function_22(x):
    x1 = x[0]
    x2 = x[1]
    if x1 == 2 or x2 == 2:
        return 1
    pi = math.pi
    ds = math.sin(pi * (x1 - 2)) * math.sin(pi * (x2 - 2))
    power_5 = abs(ds / (pi * pi * (x1 - 2) * (x2 - 2))) ** 5
    result = (1 - power_5) * (2 + (x1 - 7)**2 + 2 * (x2 - 7)**2)
    if result < 0:
        result = 0
    return result


def fitness_function_23(x):
    x1 = x[0]
    x2 = x[1]
    pi = math.pi

    power_e = math.exp(abs(100 - math.sqrt(x1**2 + x2**2) / pi))
    result = -1 * (abs(power_e * math.sin(x1) * math.sin(x2)) + 1)**-0.1
    return result


def fitness_function_24(x):
    bate = 15
    m = 5
    sum1 = 0
    sum2 = 0
    product = 1

    for i in range(len(x)):
        sum1 += (x[i] / bate)**(2 * m)
        sum2 += x[i] ** 2
        product *= math.cos(x[i]) ** 2

    result = (math.exp(-1 * sum1) - 2 * math.exp(-1 * sum2)) * product
    return result


def fitness_function_25(x):
    x1 = x[0]
    x2 = x[1]
    s1 = 1.5 - x1 + x1 * x2
    s2 = 2.25 - x1 + x1 * x2**2
    s3 = 2.625 - x1 + x1 * x2**3
    return s1**2 + s2**2 + s3**2


def fitness_function_26(x):
    bate = np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]) / 10
    c = np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                  [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                  [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                  [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
    summary = 0
    for i in range(10):
        sum1 = 0
        for j in range(len(x)):
            sum1 += (x[j] - c[j][i]) ** 2

        summary += (sum1 + bate[i]) ** -1

    return -1 * summary


def fitness_function_27(x):
    x1 = x[0]
    x2 = x[1]
    sin1 = math.sin(math.sqrt(abs(x2 + 0.5 * x1 + 47)))
    sin2 = math.sin(math.sqrt(abs(x1 - x2 - 47)))
    return -1 * (x2 + 47) * sin1 - x1 * sin2 + 959.6407


def fitness_function_28(x):
    x1 = x[0]
    x2 = x[1]
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def fitness_function_29(x):
    x1 = x[0]
    x2 = x[1]
    return (x1 + 2 * x2 - 7)**2 + (2 * x1 + x2 - 5)**2


def fitness_function_30(x):
    x1 = x[0]
    x2 = x[1]
    return 2 * x1**2 - 1.05 * x1**4 + (x1**6)/6 + x1 * x2 + x2**2









def select_fitness_function(func_idx, x):
    if func_idx == 1:
        return fitness_function_1(x)
    elif func_idx == 2:
        return fitness_function_2(x)
    elif func_idx == 3:
        return fitness_function_3(x)
    elif func_idx == 4:
        return fitness_function_4(x)
    elif func_idx == 5:
        return fitness_function_5(x)
    elif func_idx == 6:
        return fitness_function_6(x)
    elif func_idx == 7:
        return fitness_function_7(x)
    elif func_idx == 8:
        return fitness_function_8(x)
    elif func_idx == 9:
        return fitness_function_9(x)
    elif func_idx == 10:
        return fitness_function_10(x)
    elif func_idx == 11:
        return fitness_function_11(x)
    elif func_idx == 12:
        return fitness_function_12(x)





    # multimodal
    elif func_idx == 13:
        return fitness_function_13(x)
    elif func_idx == 14:
        return fitness_function_14(x)
    elif func_idx == 15:
        return fitness_function_15(x)
    elif func_idx == 16:
        return fitness_function_16(x)
    elif func_idx == 17:
        return fitness_function_17(x)
    elif func_idx == 18:
        return fitness_function_18(x)
    elif func_idx == 19:
        return fitness_function_19(x)
    elif func_idx == 20:
        return fitness_function_20(x)






    elif func_idx == 21:
        return fitness_function_21(x)
    elif func_idx == 22:
        return fitness_function_22(x)
    elif func_idx == 23:
        return fitness_function_23(x)
    elif func_idx == 24:
        return fitness_function_24(x)
    elif func_idx == 25:
        return fitness_function_25(x)
    elif func_idx == 26:
        return fitness_function_26(x)
    elif func_idx == 27:
        return fitness_function_27(x)
    elif func_idx == 28:
        return fitness_function_28(x)
    elif func_idx == 29:
        return fitness_function_29(x)
    elif func_idx == 30:
        return fitness_function_30(x)






    else:
        print("The index of function is error!!!!!!!")


# paint_image_3d()


