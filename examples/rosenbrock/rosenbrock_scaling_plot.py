import numpy as np
import matplotlib.pyplot as plt


# PySLSQP data:
dimension = np.array([100, 200, 400, 600])

distributed_time_2_subp = np.array([5.73, 8.25, 25.849, 90.312])

distributed_time_4_subp = np.array([18.27, 22.39, 28.97, 43.325])

monolithic_time = np.array([1.23, 6.95, 71.98, 342.43])

# SLSQP data:
dimension_slsqp = np.array([100, 200, 400, 600, 800, 1000])

monolithic_time_slsqp = np.array([0.12, 1.48, 10.98, 31.75, 86.71, 188.875])

distributed_time_2_subp_slsqp = np.array([0.47, 0.84, 4.41, 13.58, 32.81, 61.49])

distributed_time_4_subp_slsqp = np.array([1.05, 1.33, 1.87, 4.52, 11.45, 18.22])

distributed_time_5_subp_slsqp = np.array([1.46, 1.92, 2.27, 3.11, 7.76, 13.99])

distributed_time_10_subp_slsqp = np.array([2.20, 3.46, 4.09, 4.50, 5.96, 6.72])

plt.figure(figsize=(5,4))

# plt.semilogy(dimension, monolithic_time, marker='o', label='Monolithic', color='tab:blue')
# plt.semilogy(dimension, distributed_time_2_subp, marker='s', label='Distributed (2 subproblems)', color='tab:orange')
# plt.semilogy(dimension, distributed_time_4_subp, marker='^', label='Distributed (4 subproblems)', color='tab:green')
plt.semilogy(dimension_slsqp, monolithic_time_slsqp, marker='o', label='Monolithic SLSQP', color='tab:blue')
plt.semilogy(dimension_slsqp, distributed_time_2_subp_slsqp, marker='s', label='Distributed SLSQP (2 subproblems)', color='tab:orange')
plt.semilogy(dimension_slsqp, distributed_time_4_subp_slsqp, marker='^', label='Distributed SLSQP (4 subproblems)', color='tab:green')
plt.semilogy(dimension_slsqp, distributed_time_5_subp_slsqp, marker='v', label='Distributed SLSQP (5 subproblems)', color='tab:red')
plt.semilogy(dimension_slsqp, distributed_time_10_subp_slsqp, marker='x', label='Distributed SLSQP (10 subproblems)', color='tab:purple')

plt.xlabel('Dimension (n)')
plt.ylabel('Optimization time (s)')
plt.grid(color='lavender')
plt.legend()
# plt.savefig('scaling.pdf', bbox_inches='tight')
plt.show()