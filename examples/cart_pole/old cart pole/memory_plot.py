import numpy as np
import matplotlib.pyplot as plt



copies = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

# monolithic_memory_csdl = np.array([53.302, 73.99, 100.646, 132.946, 170.566, 213.768, 262.973, 318.20, 377.307])
monolithic_memory_jax = np.array([11.063, 21.581, 35.208, 52.54, 73.571, 98.296, 126.656, 158.705, 193.573])

distributed_memory = np.array([5.006, 5.116, 5.33, 5.497, 5.641, 5.864, 6.134, 6.368, 6.657])




# plt.figure(figsize=(4,2.5))
plt.figure(figsize=(3,3))

plt.semilogy(copies, monolithic_memory_jax, label='Monolithic', marker='o', markersize=7, linewidth=2, color='tab:purple')
plt.semilogy(copies, distributed_memory, label='ALBCD', marker='s', markersize=7, linewidth=2, color='tab:orange')
plt.xlabel('N')
plt.ylabel('Memory (MB)')
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

plt.savefig('cart_pole_memory.pdf', transparent=True, bbox_inches='tight')
plt.show()