import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize']

fig, ax = plt.subplots(1, 3, figsize=(8, 2), constrained_layout=True)
# fig, ax = plt.subplots(1, 2, figsize=(6, 2.3), constrained_layout=True)
ax = ax.flatten()


data = np.load('examples/consensus_aero_struct/consensus_aero_struct_albcd_solution.npz')
crm_error = data['error']
# print(len(crm_error))
feasibility = data['feasibility'][1:]
# print(len(feasibility))
multipliers = data['y_history']

ax[0].semilogy(crm_error, linewidth=2, color='tab:blue')
ax[0].set_xlabel('Inner-loop iteration')
ax[0].set_ylabel('Relative error $\\Vert (x - x*) / x* \\Vert$')
ax[0].grid(alpha=0.2, axis='y')
ax[0].set_xlim([0, len(crm_error) - 1])

ax[1].semilogy(feasibility, linewidth=2, color='tab:orange')
ax[1].set_xlabel('Outer-loop iteration')
ax[1].set_ylabel('Feasibility $\\Vert c \\Vert_\infty$')
ax[1].grid(alpha=0.2, axis='y')
ax[1].set_xlim([0, len(feasibility) - 1])

ax[2].plot(multipliers, linewidth=2)
# ax[2].plot(multipliers, linewidth=2, color='tab:green')
ax[2].set_xlabel('Outer-loop iteration')
ax[2].set_ylabel('Lagrange multipliers')
ax[2].grid(alpha=0.2, axis='y')
ax[2].set_xlim([0, len(multipliers) - 1])

plt.savefig('examples/consensus_aero_struct/consensus_aero_struct_albcd_solution.pdf', bbox_inches='tight')
plt.show()