import numpy as np
import matplotlib.pyplot as plt

UCSD_NAVY = "#182B49"
UCSD_BLUE = "#00629B"
UCSD_GOLD = "#C69214"
UCSD_MAGENTA = '#D462AD'
UCSD_ORANGE = '#FC8900'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize']

fig, ax = plt.subplots(1, 2, figsize=(5.33, 2.5), constrained_layout=True)
ax = ax.flatten()



ucp_error_data = np.load('examples/cart_pole/new cart pole/ucp_error_data.npz')
error_N2 = ucp_error_data['error_N2']
error_N3 = ucp_error_data['error_N3']
error_N4 = ucp_error_data['error_N4']
ax[0].semilogy(error_N2, linewidth=2, label='N=2', color=UCSD_BLUE)
ax[0].semilogy(error_N3, linewidth=2, label='N=3', color=UCSD_GOLD)
ax[0].semilogy(error_N4, linewidth=2, label='N=4', color=UCSD_MAGENTA)
ax[0].legend()
ax[0].set_xlabel('Inner-loop iteration')
ax[0].set_ylabel('Relative error')
ax[0].grid(alpha=0.2)

ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

# uncertain cart pole
ucp_data_N2 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N2_V2.npz')
feas_N2 = ucp_data_N2['feasibility'][1:]
ucp_data_N3 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N3_V2.npz')
feas_N3 = ucp_data_N3['feasibility'][1:]
ucp_data_N4 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N4_V2.npz')
feas_N4 = ucp_data_N4['feasibility'][1:]
ax[1].semilogy(feas_N2, linewidth=2, label='N=2', color=UCSD_BLUE)
ax[1].semilogy(feas_N3, linewidth=2, label='N=3', color=UCSD_GOLD)
ax[1].semilogy(feas_N4, linewidth=2, label='N=4', color=UCSD_MAGENTA)
ax[1].legend()
ax[1].set_xlabel('Outer-loop iteration')
ax[1].set_ylabel('Feasibility')
ax[1].grid(alpha=0.2)

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)


plt.savefig('ucp_convergence.pdf', bbox_inches='tight')
plt.show()