import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(3, 3))



N = 2
solution_N2 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_solution_N2.npz')
l_star = solution_N2['l']
mp_star = solution_N2['mp']
x_star = np.array(solution_N2['x_list'])
u_star = np.array(solution_N2['u_list'])

params = np.broadcast_to([l_star, mp_star], (N, 2))
solution_N2 = np.concatenate((params, x_star, u_star), axis=1).ravel()

data_N2 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N2.npz')
h2 = np.array(data_N2['history'])

# error = np.zeros(len(h2))
# for i in range(len(h2)):
#     h_i = h2[i].flatten()
#     error_i = np.linalg.norm((h_i - solution_N2))
#     error[i] = error_i
error = np.linalg.norm(h2.reshape(len(h2), -1) - solution_N2, axis=1)

plt.semilogy(error, linewidth=2, label='N=2')
# plt.grid()
# plt.show()





N = 3
solution_N3 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_solution_N3.npz')
l_star = solution_N3['l']
mp_star = solution_N3['mp']
x_star = np.array(solution_N3['x_list'])
u_star = np.array(solution_N3['u_list'])

params = np.broadcast_to([l_star, mp_star], (N, 2))
solution_N3 = np.concatenate((params, x_star, u_star), axis=1).ravel()

data_N3 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N3.npz')
h3 = np.array(data_N3['history'])

error = np.linalg.norm(h3.reshape(len(h3), -1) - solution_N3, axis=1)

plt.semilogy(error, linewidth=2, label='N=3')
# plt.grid()
# plt.show()





N = 4
solution_N4 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_solution_N4.npz')
l_star = solution_N4['l']
mp_star = solution_N4['mp']
x_star = np.array(solution_N4['x_list'])
u_star = np.array(solution_N4['u_list'])

params = np.broadcast_to([l_star, mp_star], (N, 2))
solution_N4 = np.concatenate((params, x_star, u_star), axis=1).ravel()

data_N4 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N4.npz')
h4 = np.array(data_N4['history'])

error = np.linalg.norm(h4.reshape(len(h4), -1) - solution_N4, axis=1)

plt.semilogy(error, linewidth=2, label='N=4')




plt.title('Uncertain Cart Pole Co-Design')
plt.ylabel('Error')
plt.xlabel('Iteration')
plt.grid(alpha=0.2)
plt.legend()

plt.savefig('examples/cart_pole/new cart pole/error_plot.pdf', bbox_inches='tight')
plt.show()