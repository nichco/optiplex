import numpy as np
import matplotlib.pyplot as plt


monolithic_N = np.array([2, 4, 6, 8, 10])
monolithic_memory = np.array([188.9, 610.8, 1276.6, 2186.3, 3339.9])

distributed_N = np.array([2, 4, 6, 8, 10])
distributed_memory = np.array([100.3, 129.6, 173.5, 204.2, 235.0])



plt.figure(figsize=(4,3))

plt.semilogy(monolithic_N, monolithic_memory, marker='o', linewidth=1, label='Monolithic')
plt.semilogy(distributed_N, distributed_memory, marker='s', linewidth=1, label='Distributed')

plt.xlabel('Number of Subproblems')
plt.ylabel('Memory (MB)')

plt.legend(loc='upper left', fontsize=10)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_position(('outward', 6))
ax.spines['left'].set_position(('outward', 6))

plt.savefig('bwb_memory_scaling.pdf', bbox_inches='tight')
plt.show()