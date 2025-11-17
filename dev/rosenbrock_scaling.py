import numpy as np
import matplotlib.pyplot as plt



dimension = np.array([100, 200, 400, 600])

distributed_time_2_subp = np.array([5.73, 8.25, 25.849, 90.312])

distributed_time_4_subp = np.array([18.27, 22.39, 28.97, 43.325])

monolithic_time = np.array([1.23, 6.95, 71.98, 342.43])


plt.figure(figsize=(5,4))

plt.semilogy(dimension, monolithic_time, marker='o', label='Monolithic SLSQP', color='tab:blue')
plt.semilogy(dimension, distributed_time_2_subp, marker='s', label='Distributed SLSQP 2', color='tab:orange')
plt.semilogy(dimension, distributed_time_4_subp, marker='^', label='Distributed SLSQP 4', color='tab:green')

# add the rosenbrock function as text
plt.text(0.5, 0.1, r'$f(x) = \sum_{i=1}^{n-1} \left[100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right]$', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)

plt.xlabel('Problem Dimension (n)')
plt.ylabel('Wall Time (s)')
plt.grid(color='lavender')
plt.legend()
plt.savefig('scaling.png', bbox_inches='tight', dpi=300, transparent=True)
plt.show()