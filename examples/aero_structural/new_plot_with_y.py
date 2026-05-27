import numpy as np
import matplotlib.pyplot as plt

data = np.load('examples/aero_structural/data/history4_with_y.npz')

time = data['x_time']
error = data['error']
feasibility = data['feasibility']
mu = data['mu_history']
y = data['multipliers']

plt.figure(figsize=(4, 3))
plt.semilogy(time, error, color='tab:blue', label='Error', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.grid(axis='y', color='lavender')
# plt.savefig('simple_aero_struct_error_2.pdf', bbox_inches='tight')
plt.show()




plt.figure(figsize=(4, 3))
plt.semilogy(feasibility, color='tab:blue', label='Feasibility', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Feasibility')
plt.grid(axis='y', color='lavender')
# plt.savefig('simple_aero_struct_feasibility.pdf', bbox_inches='tight')
plt.show()


plt.figure(figsize=(4, 3))
y = abs(np.asarray(y))
y = y[1:,:] # skip the first iteration which is all zeros
x = np.arange(y.shape[0]) + 1 # iteration numbers starting from 1
# print('y shape: ', y.shape)
# exit()
for i in range(y.shape[1]):
    plt.semilogy(x, y[:, i], label=f'y[{i}]', linewidth=2, linestyle='-')
# plt.legend()
plt.xlabel('Outer-loop iteration')
plt.ylabel('Absolute Lagrange multipliers')
plt.grid(axis='y', color='lavender')
# plt.savefig('simple_aero_struct_y.pdf', bbox_inches='tight')
plt.show()




plt.figure(figsize=(4, 3))
mu = np.asarray(mu)
for i in range(mu.shape[1]):
    plt.semilogy(time, mu[:, i], label=f'mu[{i}]', linewidth=2, linestyle='-')
# plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Penalty parameters')
plt.grid(axis='y', color='lavender')
# plt.savefig('simple_aero_struct_mu_2.pdf', bbox_inches='tight')
plt.show()