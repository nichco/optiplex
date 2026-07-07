import numpy as np
import matplotlib.pyplot as plt



subproblems = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

# monolithic_time = np.array([11.626, 42.037, 133.012, 621.379, 1467.451, 2944.026, 6370])
monolithic_time = np.array([9.86, 24.22, 112.345, 386.23, 755.009, 1504.773, 2785.930, 3508.666, 15755.439])

# distributed_time = np.array([52.335, 88.211, 205.231, 161.269, 395.084, 429.445, 578.767])
distributed_time = np.array([40.675, 58.815, 176.175, 265.7, 320.679, 399.415, 565.788, 588.668, 817.812])

# plt.figure(figsize=(4,2.5))
plt.figure(figsize=(3,3))

plt.semilogy(subproblems, monolithic_time, label='Monolithic', marker='o', markersize=7, linewidth=2, color='tab:purple', mec='k')
plt.semilogy(subproblems, distributed_time, label='Distributed', marker='s', markersize=7, linewidth=2, color='tab:orange', mec='k')
plt.xlabel('N')
plt.ylabel('Time (s)')
plt.legend()
# plt.grid(True, linewidth=0.5, alpha=0.3)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_position(('outward', 6))
ax.spines['left'].set_position(('outward', 6))


plt.title('Uncertain Cart Pole Co-Design')
plt.grid(color='blue', alpha=0.1, axis='y')

plt.savefig('cart_pole_timing.pdf', transparent=True, bbox_inches='tight')
plt.show()