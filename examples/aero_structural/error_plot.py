import numpy as np
import matplotlib.pyplot as plt

# data = np.load('examples/aero_structural/history_idf.npz')
data = np.load('examples/aero_structural/history_idf_with_feas.npz')

time = data['x_time']
error = data['error']
mu = data['mu_history']

# fig, ax1 = plt.subplots(figsize=(4.5, 3))

# color_error = 'tab:blue'
# ax1.set_xlabel('Wall time (s)')
# ax1.set_ylabel('Error', color=color_error)
# ax1.semilogy(time, error, color=color_error, label='Error', linewidth=2)
# ax1.tick_params(axis='y', labelcolor=color_error)

# ax2 = ax1.twinx()
# color_mu = 'tab:orange'
# ax2.set_ylabel('Penalty parameter', color=color_mu)
# ax2.semilogy(time, mu, color=color_mu, label='μ', linewidth=2)
# ax2.tick_params(axis='y', labelcolor=color_mu)

# fig.tight_layout()


plt.figure(figsize=(4, 3))
plt.semilogy(time, error, color='tab:blue', label='Error', linewidth=2)
plt.xlabel('Wall time (s)')
plt.ylabel('Error')
plt.grid(axis='y', color='lavender')

plt.savefig('simple_aero_struct_convergence.pdf', bbox_inches='tight')
plt.show()