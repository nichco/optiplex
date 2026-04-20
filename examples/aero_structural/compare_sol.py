import numpy as np


solution = np.load('examples/aero_structural/solution.npz')
x_star = np.concatenate([solution['twist'], solution['thickness']])

new_solution = np.load('examples/aero_structural/new_solution.npz')
x_new = np.concatenate([new_solution['twist'], new_solution['thickness']])



error = np.linalg.norm((x_new - x_star) / x_star)
print('Relative error between new solution and reference solution: ', error)