import numpy as np
import matplotlib.pyplot as plt


# n=200
N = np.array([2, 4, 8, 10])
decomp_n200_time = np.array([62.14, 141.3, 111.4, 158.2])

decomp_n200_itr = np.array([7, 20, 40, 48])

distributed_n200_time = np.array([8.31, 16.78, 39.18, 44.98])

distributed_n200_itr = np.array([41, 105, 233, 297])


plt.figure(figsize=(5,4))

plt.plot(N, decomp_n200_time, marker='o', label='With automatic decomposition')
plt.plot(N, distributed_n200_time, marker='o', label='Fixed decomposition')

# add the rosenbrock function as text
plt.text(0.45, 0.9, r'$f(x) = \sum_{i=1}^{200-1} \left[100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right]$', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)

plt.xlabel('Number of subproblems')
plt.ylabel('Wall time (s)')
plt.grid(color='lavender', alpha=0.5)
plt.legend()
plt.savefig('decomp_time.png', bbox_inches='tight', dpi=300, transparent=True)
plt.show()


plt.figure(figsize=(5,4))

plt.plot(N, decomp_n200_itr, marker='o', label='With automatic decomposition')
plt.plot(N, distributed_n200_itr, marker='o', label='Fixed decomposition')

# add the rosenbrock function as text
# plt.text(0.45, 0.1, r'$f(x) = \sum_{i=1}^{200-1} \left[100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right]$', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes)

plt.xlabel('Number of subproblems')
plt.ylabel('Number of iterations')
plt.grid(color='lavender', alpha=0.5)
plt.legend()
plt.savefig('decomp_itr.png', bbox_inches='tight', dpi=300, transparent=True)
plt.show()