import numpy as np
import matplotlib.pyplot as plt



dimension = np.array([100, 200, 400, 600, 800, 1000])

monolithic_memory = np.array([3.37, 5.94, 16.12, 33.01, 56.63, 86.96])

distributed_memory_2_subp = np.array([2.91, 3.57, 6.14, 10.40, 16.35, 23.95])

distributed_memory_5_subp = np.array([2.81, 2.93, 3.34, 4.05, 5.00, 6.25])

distributed_memory_10_subp = np.array([2.98, 2.99, 3.1, 3.3, 3.53, 3.85])


plt.figure(figsize=(4,3))
# plt.rcParams.update({'font.size': 12})

plt.semilogy(dimension, monolithic_memory, marker='o', label='Monolithic SLSQP', linewidth=1, markersize=8, color='tab:grey')
plt.semilogy(dimension, distributed_memory_2_subp, marker='s', label='Dist. SLSQP (2 subproblems)', linewidth=1, markersize=8, color='tab:blue')
plt.semilogy(dimension, distributed_memory_5_subp, marker='^', label='Dist. SLSQP (5 subproblems)', linewidth=1, markersize=8, color='tab:blue')
plt.semilogy(dimension, distributed_memory_10_subp, marker='v', label='Dist. SLSQP (10 subproblems)', linewidth=1, markersize=8, color='tab:blue')

plt.xlabel("Dimension (n)")
plt.ylabel("Memory (MB)")
plt.legend(fontsize=10)
# plt.grid(color='lavender')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_position(('outward', 0))
ax.spines['left'].set_position(('outward', 6))

ax.set_xticks([100, 500, 1000])

# plt.savefig('rosenbrock_memory_scaling.pdf', bbox_inches='tight')
plt.show()