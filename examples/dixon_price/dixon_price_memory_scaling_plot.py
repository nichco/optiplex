import numpy as np
import matplotlib.pyplot as plt



dimension = np.linspace(100, 1000, 10, dtype=int)

monolithic_memory = np.array([4.01, 6.10, 9.55, 14.36, 20.53, 28.05, 36.94, 47.19, 71.00, 87.01])

distributed_memory_2_subp = np.array([2.64, 3.18, 4.07, 5.29, 6.85, 8.75, 11.00, 13.58, 16.50, 19.76])

distributed_memory_5_subp = np.array([2.52, 2.64, 2.77, 2.98, 3.24, 3.56, 3.92, 4.35, 4.84, 5.37])

distributed_memory_10_subp = np.array([2.58, 2.60, 2.62, 2.71, 2.74, 2.83, 2.93, 3.05, 3.17, 3.31])


plt.figure(figsize=(4,3))
# plt.rcParams.update({'font.size': 12})

plt.semilogy(dimension, monolithic_memory, marker='o', label='Monolithic SLSQP', linewidth=2, markersize=8, color='tab:blue')
plt.semilogy(dimension, distributed_memory_2_subp, marker='s', label='Dist. SLSQP (2 subproblems)', linewidth=2, markersize=8, color='tab:orange')
plt.semilogy(dimension, distributed_memory_5_subp, marker='^', label='Dist. SLSQP (5 subproblems)', linewidth=2, markersize=8, color='tab:green')
plt.semilogy(dimension, distributed_memory_10_subp, marker='v', label='Dist. SLSQP (10 subproblems)', linewidth=2, markersize=8, color='tab:purple')

plt.xlabel("Dimension (n)")
plt.ylabel("Memory (MB)")
plt.legend(fontsize=10)
plt.grid(color='lavender')

plt.savefig('dixon_price_memory_scaling.pdf', bbox_inches='tight')
plt.show()