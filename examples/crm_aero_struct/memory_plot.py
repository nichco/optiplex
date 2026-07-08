import numpy as np
import matplotlib.pyplot as plt




# with full model evals
# N_d = np.array([2, 3, 4, 6, 8])
# ALBCD_mem = np.array([16.50, 17.71, 19.71, 23.015, 26.7])

# with reduced model evals
N_d = np.array([2, 3, 4, 6, 8])
ALBCD_mem = np.array([16.50, 17.71, 19.71, 23.015, 26.7])

N_m = np.array([2, 4, 6, 8, 10])
monolithic_mem = np.array([14.61, 21.81, 28.81, 36.23, 43.2])


plt.figure(figsize=(3,3))

plt.plot(N_d, ALBCD_mem, marker='o', label='ALBCD', color='tab:orange')
plt.plot(N_m, monolithic_mem, marker='o', label='Monolithic', color='tab:purple')
# plt.semilogy(N_d, ALBCD_mem, marker='o', label='ALBCD')
# plt.semilogy(N_m, monolithic_mem, marker='o', label='Monolithic')

plt.xlabel('N')
plt.ylabel('Memory (MB)')
plt.legend()

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_position(('outward', 6))
ax.spines['left'].set_position(('outward', 6))


plt.title('Multi-Point CRM')
plt.grid(color='blue', alpha=0.1, axis='y')

plt.savefig('crm_memory.pdf', transparent=True, bbox_inches='tight')
plt.show()