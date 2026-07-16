import numpy as np
import matplotlib.pyplot as plt



n = np.array([100, 200, 400, 600, 800, 1000])

mean_time_N2 = np.array([0.78, 1.16, 4.89, 12.81, 36.04, 64.94])
std_time_N2 = np.array([0.24, 0.28, 1.12, 3.23, 9.89, 19.59])

mean_time_N5 = np.array([2.90, 3.19, 4.12, 2.93, 4.55, 9.25])
std_time_N5 = np.array([0.85, 1.02, 1.25, 0.71, 0.91, 2.39])


fig, ax = plt.subplots(figsize=(4, 3))

# plt.semilogy(n, mean_time_N2, marker='o', color='tab:blue', label='N = 2')
plt.semilogy(n, mean_time_N2, color='tab:blue', label='N = 2')
plt.fill_between(n, mean_time_N2 - std_time_N2, mean_time_N2 + std_time_N2,
                 color='tab:blue', alpha=0.2)

# plt.semilogy(n, mean_time_N5, marker='o', color='tab:orange', label='N = 5')
plt.semilogy(n, mean_time_N5, color='tab:orange', label='N = 5')
plt.fill_between(n, mean_time_N5 - std_time_N5, mean_time_N5 + std_time_N5,
                 color='tab:orange', alpha=0.2)

plt.xlim(100, 1000)
plt.xlabel('n')
plt.ylabel('Time')
plt.legend()
fig.tight_layout()
plt.show()