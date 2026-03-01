import numpy as np
import matplotlib.pyplot as plt


monolithic_N = np.array([2, 4, 6, 8, 20])
monolithic_times = np.array([94.2, 263.8, 580.6, 762.9, 5162.2])

distributed_N = np.array([2, 4, 6, 8, 10])
distributed_times = np.array([298.4, 534.8, 937.5, 1551.2, 2227.4])



plt.semilogy(monolithic_N, monolithic_times, marker='o', label='Monolithic')
plt.semilogy(distributed_N, distributed_times, marker='o', label='Distributed')

plt.xlabel('Number of Subproblems')
plt.ylabel('Time (s)')

plt.show()