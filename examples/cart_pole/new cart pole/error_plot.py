import numpy as np
import matplotlib.pyplot as plt

N = 2
n = 30

solution = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_solution_N2.npz')
l_star = solution['l']
mp_star = solution['mp']
x_list_star = np.array(solution['x_list'])
u_list_star = np.array(solution['u_list'])

v_stars = []
for i in range(N):
    v_i_star = np.concatenate((np.array([l_star, mp_star]), x_list_star[i], u_list_star[i]))
    v_stars.append(v_i_star)

solution = np.concatenate(v_stars)

data = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N2.npz')
h = data['history']
h = np.array(h)
n_itr = len(h)

error = np.zeros(n_itr)
for i in range(n_itr):
    h_i = h[i].flatten()
    error_i = np.linalg.norm((h_i - solution))
    error[i] = error_i

plt.semilogy(error, linewidth=2)
plt.grid()
plt.show()