import numpy as np
import matplotlib.pyplot as plt

data = np.load('examples/aero_structural/history4_r1p5.npz')

time = data['x_time']
error = data['error']
feasibility = data['feasibility']
mu = data['mu_history']

plt.figure(figsize=(4, 3))
plt.semilogy(time, error, color='tab:blue', label='Error', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.grid(axis='y', color='lavender')
plt.savefig('simple_aero_struct_error_2.pdf', bbox_inches='tight')
plt.show()




plt.figure(figsize=(4, 3))
plt.semilogy(feasibility, color='tab:blue', label='Feasibility', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Feasibility')
plt.grid(axis='y', color='lavender')
plt.savefig('simple_aero_struct_feasibility.pdf', bbox_inches='tight')
plt.show()




plt.figure(figsize=(4, 3))
mu = np.asarray(mu)
for i in range(mu.shape[1]):
    plt.semilogy(time, mu[:, i], label=f'mu[{i}]', linewidth=2, linestyle='-')
# plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Penalty parameters')
plt.grid(axis='y', color='lavender')
plt.savefig('simple_aero_struct_mu_2.pdf', bbox_inches='tight')
plt.show()