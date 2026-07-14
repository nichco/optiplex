import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize']

fig, ax = plt.subplots(1, 2, figsize=(5.3333, 2.5), constrained_layout=True)
ax = ax.flatten()


# UCP
N = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
monolithic_mem = np.array([11.063, 21.581, 35.208, 52.54, 73.571, 98.296, 126.656, 158.705, 193.573])
ALBCD_mem = np.array([5.006, 5.116, 5.33, 5.497, 5.641, 5.864, 6.134, 6.368, 6.657])

ax[0].plot(N, ALBCD_mem, marker='o', label='ALBCD', color='tab:orange')
ax[0].plot(N, monolithic_mem, marker='o', label='Monolithic', color='tab:purple')
ax[0].set_xlabel('N')
ax[0].set_ylabel('Memory (MB)')
ax[0].legend()
ax[0].set_title('Uncertain Cart Pole')
ax[0].grid(color='blue', alpha=0.1, axis='y')


# MP CRM
N = np.array([2, 4, 6, 8, 10])
ALBCD_mem = np.array([13.19, 15.00, 16.19, 17.17, 20.48])
monolithic_mem = np.array([14.61, 21.81, 28.81, 36.23, 43.2])

ax[1].plot(N, ALBCD_mem, marker='o', label='ALBCD', color='tab:orange')
ax[1].plot(N, monolithic_mem, marker='o', label='Monolithic', color='tab:purple')
ax[1].set_xlabel('N')
ax[1].set_ylabel('Memory (MB)')
ax[1].legend()
ax[1].set_title('Multi-Point CRM')
ax[1].grid(color='blue', alpha=0.1, axis='y')


plt.savefig('memory_plot.pdf', bbox_inches='tight')
plt.show()