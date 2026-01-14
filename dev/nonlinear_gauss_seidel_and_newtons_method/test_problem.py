import numpy as np
import math
import matplotlib.pyplot as plt

# def f(x):
#     x1, x2 = x
#     r1 = 0.5 * (x1 - 1) + (1/6) * (x2 - 2) + (1/8) * (x1 - 1)**2 + 2 * (1/24) * (x1 - 1) * (x2 - 2) + (1/72) * (x2 - 2)**2
#     r2 = (1/6) * (x1 - 1) + 0.5 * (x2 - 2) + (1/72) * (x1 - 1)**2 + 2 * (1/24) * (x1 - 1) * (x2 - 2) + (1/8) * (x2 - 2)**2
#     return np.array([r1, r2])




# f(r) = r^-1
# g(i, j_k) = 1.0001**(-np.abs(i - j_k))
# alpha in the range of 0.5 to 1.5
# set x0 randomly in the range -150 to 150, excluding zero

n = 5

def fun(x):

    f = np.zeros(n)
    for i in range(n):
        xi = x[i]
        f[i] = xi**(-1)
    
    return f




def f(x, d):


    for i in range(n):
        d_i = d[i]
        xi = x[i]

        term_1 = sum([1 / math.factorial(r) for r in range(1, d_i)])

    return