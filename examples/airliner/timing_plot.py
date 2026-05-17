import numpy as np
import matplotlib.pyplot as plt


monolithic_N = np.array([2, 4, 6, 8])
monolithic_times = np.array([189.4, 1084.4, 3755.1, 8765.8])

distributed_N = np.array([2, 4, 6, 8, 10])
distributed_times = np.array([99.3, 145.9, 194.8, 284.0, 343.8])



plt.figure(figsize=(4,2.5))

plt.semilogy(monolithic_N, monolithic_times, marker='o', linewidth=1, label='Monolithic', color='tab:purple', mec='k')
plt.semilogy(distributed_N, distributed_times, marker='s', linewidth=1, label='Distributed', color='tab:orange', mec='k')

plt.xlabel('Number of Subproblems')
plt.ylabel('Time (s)')

plt.legend(loc='upper left', fontsize=10)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_position(('outward', 6))
ax.spines['left'].set_position(('outward', 6))

plt.grid(color='blue', alpha=0.1, axis='y')

plt.savefig('bwb_time_scaling.pdf', bbox_inches='tight')
plt.show()