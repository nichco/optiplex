import numpy as np
import matplotlib.pyplot as plt

data_m6 = np.load('examples/aero_structural/history_idf_mu6.npz')
data_m8 = np.load('examples/aero_structural/history_idf_mu8.npz')
data_m10 = np.load('examples/aero_structural/history_idf_mu10.npz')
data_m12 = np.load('examples/aero_structural/history_idf_mu12.npz')
data_m15 = np.load('examples/aero_structural/history_idf_mu15.npz')


time_m6 = data_m6['x_time']
error_m6 = data_m6['error']
mu_m6 = data_m6['mu_history']
time_m8 = data_m8['x_time']
error_m8 = data_m8['error']
mu_m8 = data_m8['mu_history']
time_m10 = data_m10['x_time']
error_m10 = data_m10['error']
mu_m10 = data_m10['mu_history']
time_m12 = data_m12['x_time']
error_m12 = data_m12['error']
mu_m12 = data_m12['mu_history']
time_m15 = data_m15['x_time']
error_m15 = data_m15['error']
mu_m15 = data_m15['mu_history']

fig, ax1 = plt.subplots(figsize=(4.5, 3))

ax1.set_xlabel('Wall time (s)')
ax1.set_ylabel('Error')

ax1.semilogy(time_m6, error_m6, color='tab:blue', label='μ0=6', linewidth=2)
ax1.semilogy(time_m8, error_m8, color='tab:orange', label='μ0=8', linewidth=2)
ax1.semilogy(time_m10, error_m10, color='tab:red', label='μ0=10', linewidth=2)
ax1.semilogy(time_m12, error_m12, color='tab:green', label='μ0=12', linewidth=2)
# ax1.semilogy(time_m15, error_m15, color='tab:blue', linestyle='--', label='μ0=15', linewidth=2)
ax1.legend(loc='upper right')
ax1.tick_params(axis='y')

# ax2 = ax1.twinx()
# color_mu = 'tab:orange'
# ax2.set_ylabel('Penalty parameter', color=color_mu)
# ax2.semilogy(time_m6, mu_m6, color=color_mu, label='μ', linewidth=2)
# ax2.semilogy(time_m8, mu_m8, color=color_mu, label='μ', linewidth=2)
# ax2.semilogy(time_m10, mu_m10, color=color_mu, label='μ', linewidth=2)
# ax2.semilogy(time_m12, mu_m12, color='tab:orange', linestyle='-.', label='...', linewidth=2)
# # ax2.semilogy(time_m15, mu_m15, color='tab:orange', linestyle='--', label='...', linewidth=2)
# # ax2.legend(loc='upper left')

# ax2.tick_params(axis='y', labelcolor=color_mu)

fig.tight_layout()

plt.savefig('simple_aero_struct_mu.pdf', bbox_inches='tight')
plt.show()