import numpy as np
import matplotlib.pyplot as plt

UCSD_NAVY = "#182B49"
UCSD_BLUE = "#00629B"
UCSD_GOLD = "#C69214"

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize']

fig, ax = plt.subplots(1, 3, figsize=(6, 2), constrained_layout=True)
# fig, ax = plt.subplots(1, 2, figsize=(6, 2.3), constrained_layout=True)
ax = ax.flatten()


data = np.load('examples/consensus_aero_struct/consensus_aero_struct_albcd_solution.npz')
crm_error = data['error']
# print(len(crm_error))
feasibility = data['feasibility'][1:]
# print(len(feasibility))
multipliers = data['y_history']

ax[0].semilogy(crm_error, linewidth=1.5, color=UCSD_BLUE)
ax[0].set_xlabel('Inner-loop iteration')
# ax[0].set_ylabel('Relative error $\\Vert (x - x*) / x* \\Vert$')
ax[0].set_ylabel('Relative error')
ax[0].grid(alpha=0.2, axis='y')
ax[0].set_xlim([0, len(crm_error) - 1])
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

ax[1].semilogy(feasibility, linewidth=1.5, color=UCSD_GOLD)
ax[1].set_xlabel('Outer-loop iteration')
ax[1].set_ylabel('Feasibility')
ax[1].grid(alpha=0.2, axis='y')
ax[1].set_xlim([0, len(feasibility) - 1])
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

load_lines = ax[2].plot(multipliers[:, :-1], linewidth=1.5, color=UCSD_BLUE)
weight_line, = ax[2].plot(multipliers[:, -1], linewidth=1.5, color=UCSD_GOLD)
# ax[2].plot(multipliers, linewidth=2, color='tab:green')
ax[2].set_xlabel('Outer-loop iteration')
ax[2].set_ylabel('Lagrange multipliers')
ax[2].grid(alpha=0.2, axis='y')
ax[2].set_xlim([0, len(multipliers) - 1])
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
legend = ax[2].legend(
	handles=[load_lines[0], weight_line],
	labels=['Load multipliers', 'Weight multiplier'],
	loc='upper left',
	# frameon=False,
	fontsize=9,
	handlelength=0,
	handletextpad=0,
)
for text, color in zip(legend.get_texts(), [UCSD_BLUE, UCSD_GOLD]):
	text.set_color(color)

plt.savefig('examples/consensus_aero_struct/consensus_aero_struct_albcd_solution.pdf', bbox_inches='tight')
plt.show()