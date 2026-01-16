import numpy as np
import matplotlib.pyplot as plt



dimension = np.array([100, 200, 400, 600, 800, 1000])

distributed_time_2_subp = np.array([6.207, 9.175, 16.641, 24.015, 34.623, 43.028])

distributed_time_4_subp = np.array([14.370, 29.073, 30.941, 36.024, 50.109, 62.325])

monolithic_time = np.array([1.7921, 5.199, 11.377, 22.4306, 41.835, 65.757])


plt.figure(figsize=(5,4))

plt.semilogy(dimension, monolithic_time, marker='o', label='Monolithic IPOPT', color='tab:blue')
plt.semilogy(dimension, distributed_time_2_subp, marker='s', label='Distributed IPOPT (2 subproblems)', color='tab:orange')
plt.semilogy(dimension, distributed_time_4_subp, marker='^', label='Distributed IPOPT (4 subproblems)', color='tab:green')

plt.xlabel('Dimension (n)')
plt.ylabel('Optimization time (s)')
plt.grid(color='lavender')
plt.legend()
# plt.savefig('scaling_ipopt.png', bbox_inches='tight', dpi=300, transparent=True)
plt.show()