import numpy as np
import matplotlib.pyplot as plt

data_r11 = np.load('examples/aero_structural/history_idf_r11.npz')
data_r12 = np.load('examples/aero_structural/history_idf.npz')
data_r15 = np.load('examples/aero_structural/history_idf_r15.npz')
data_r20 = np.load('examples/aero_structural/history_idf_r20.npz')

time_r11 = data_r11['x_time']
error_r11 = data_r11['error']
mu_r11 = data_r11['mu_history']
time_r12 = data_r12['x_time']
error_r12 = data_r12['error']
mu_r12 = data_r12['mu_history']
time_r15 = data_r15['x_time']
error_r15 = data_r15['error']
mu_r15 = data_r15['mu_history']
time_r20 = data_r20['x_time']
error_r20 = data_r20['error']
mu_r20 = data_r20['mu_history']

fig, ax1 = plt.subplots(figsize=(4.5, 3))

color_error = 'tab:blue'
ax1.set_xlabel('Wall time (s)')
ax1.set_ylabel('Error', color=color_error)
ax1.semilogy(time_r11, error_r11, color=color_error, label='ρ=1.1', linewidth=2)
ax1.semilogy(time_r12, error_r12, color='tab:blue', linestyle='--', label='ρ=1.2', linewidth=2)
ax1.semilogy(time_r15, error_r15, color='tab:blue', linestyle='-.', label='ρ=1.5', linewidth=2)
ax1.semilogy(time_r20, error_r20, color='tab:blue', linestyle=':', label='ρ=2.0', linewidth=2)
# ax1.semilogy(time_r11, error_r11, color='tab:blue', label='Error', linewidth=2)
# ax1.semilogy(time_r12, error_r12, color='tab:orange', label='Error (ρ=1.2)', linewidth=2)
# ax1.semilogy(time_r15, error_r15, color='tab:green', label='Error (ρ=1.5)', linewidth=2)
# ax1.semilogy(time_r20, error_r20, color='tab:red', label='Error (ρ=2.0)', linewidth=2)
ax1.legend(loc='upper right')
ax1.tick_params(axis='y', labelcolor=color_error)

ax2 = ax1.twinx()
color_mu = 'tab:orange'
ax2.set_ylabel('Penalty parameter', color=color_mu)
ax2.semilogy(time_r11, mu_r11, color=color_mu, label='μ', linewidth=2)
ax2.semilogy(time_r12, mu_r12, color='tab:orange', linestyle='--', label='μ (ρ=1.2)', linewidth=2)
ax2.semilogy(time_r15, mu_r15, color='tab:orange', linestyle='-.', label='μ (ρ=1.5)', linewidth=2)
ax2.semilogy(time_r20, mu_r20, color='tab:orange', linestyle=':', label='μ (ρ=2.0)', linewidth=2)
# ax2.legend(loc='upper left')

ax2.tick_params(axis='y', labelcolor=color_mu)

fig.tight_layout()

plt.savefig('simple_aero_struct_rho.pdf', bbox_inches='tight')
plt.show()