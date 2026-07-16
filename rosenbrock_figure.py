import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize']

fig, ax = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)
ax = ax.flatten()




n = np.array([100, 200, 400, 600, 800, 1000])

monolithic_time = np.array([0.12, 1.48, 10.98, 31.75, 86.71, 188.875])
distributed_time_N2 = np.array([0.47, 0.84, 4.41, 13.58, 32.81, 61.49])
distributed_time_N5 = np.array([1.46, 1.92, 2.27, 3.11, 7.76, 13.99])
distributed_time_N10 = np.array([2.20, 3.46, 4.09, 4.50, 5.96, 6.72])

ax[0].semilogy(n, monolithic_time, marker='o', label='Monolithic', color='tab:gray', linewidth=1, markersize=8, mec='k')
ax[0].semilogy(n, distributed_time_N2, marker='s', label='ALBCD N=2', color='tab:blue', linewidth=1, markersize=7, mec='k')
ax[0].semilogy(n, distributed_time_N5, marker='d', label='ALBCD N=5', color='tab:orange', linewidth=1, markersize=7, mec='k')
ax[0].semilogy(n, distributed_time_N10, marker='v', label='ALBCD N=10', color='tab:green', linewidth=1, markersize=7, mec='k')
ax[0].set_xlabel('Dimension (n)')
ax[0].set_ylabel('Optimization time (s)')
ax[0].legend(loc='upper left', fontsize=10)

monolithic_memory = np.array([3.37, 5.94, 16.12, 33.01, 56.63, 86.96])
distributed_memory_2_subp = np.array([2.91, 3.57, 6.14, 10.40, 16.35, 23.95])
distributed_memory_5_subp = np.array([2.81, 2.93, 3.34, 4.05, 5.00, 6.25])
distributed_memory_10_subp = np.array([2.98, 2.99, 3.1, 3.3, 3.53, 3.85])

ax[1].semilogy(n, monolithic_memory, marker='o', label='Monolithic', color='tab:gray', linewidth=1, markersize=8, mec='k')
ax[1].semilogy(n, distributed_memory_2_subp, marker='s', label='ALBCD N=2', color='tab:blue', linewidth=1, markersize=7, mec='k')
ax[1].semilogy(n, distributed_memory_5_subp, marker='d', label='ALBCD N=5', color='tab:orange', linewidth=1, markersize=7, mec='k')
ax[1].semilogy(n, distributed_memory_10_subp, marker='v', label='ALBCD N=10', color='tab:green', linewidth=1, markersize=7, mec='k')
ax[1].set_xlabel('Dimension (n)')
ax[1].set_ylabel('Memory (MB)')
ax[1].legend(loc='upper left', fontsize=10)

dimension = np.array([100, 500, 1000])

monolithic_time_slsqp_beta_10 = np.array([0.14, 6.25, 53.60])
monolithic_time_slsqp_beta_100 = np.array([0.40, 24.33, 200.43])
monolithic_time_slsqp_beta_200 = np.array([0.58, 31.71, 301.34])
monolithic_time_slsqp_beta_300 = np.array([0.52, 41.66, 344.51])

distributed_time_5_subp_slsqp_beta_10 = np.array([0.66, 1.36, 5.116])
distributed_time_5_subp_slsqp_beta_100 = np.array([4.62, 8.16, 24.61])
distributed_time_5_subp_slsqp_beta_200 = np.array([11.98, 15.76, 40.94])
distributed_time_5_subp_slsqp_beta_300 = np.array([13.95, 24.06, 56.19])

delta_b10 = distributed_time_5_subp_slsqp_beta_10 / monolithic_time_slsqp_beta_10
delta_b100 = distributed_time_5_subp_slsqp_beta_100 / monolithic_time_slsqp_beta_100
delta_b200 = distributed_time_5_subp_slsqp_beta_200 / monolithic_time_slsqp_beta_200
delta_b300 = distributed_time_5_subp_slsqp_beta_300 / monolithic_time_slsqp_beta_300

ax[2].semilogy(dimension, delta_b10, marker='o', label=r'$\beta=10$', color='tab:red', linewidth=1, markersize=7, mec='k')
ax[2].semilogy(dimension, delta_b100, marker='o', label=r'$\beta=100$', color='tab:blue', linewidth=1, markersize=7, mec='k')
ax[2].semilogy(dimension, delta_b200, marker='s', label=r'$\beta=200$', color='tab:orange', linewidth=1, markersize=7, mec='k')
ax[2].semilogy(dimension, delta_b300, marker='d', label=r'$\beta=300$', color='tab:green', linewidth=1, markersize=7, mec='k')
ax[2].set_xlabel('Dimension (n)')
ax[2].set_ylabel('Distributed Time / Monolithic Time')
ax[2].legend(loc='upper right', fontsize=10)

ax[2].axhline(y=1, color='tab:gray', linestyle='--', linewidth=1)
ax[2].text(1.0, 0.45, 'Monolithic is faster', transform=plt.gca().transAxes, ha='right', va='center', fontsize=8, color='gray')
ax[2].text(1.0, 0.39, 'Distributed is faster', transform=plt.gca().transAxes, ha='right', va='center', fontsize=8, color='gray')


plt.savefig('rosenbrock_plot.pdf', bbox_inches='tight')
plt.show()