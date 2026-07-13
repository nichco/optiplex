import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize']

fig, ax = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)
ax = ax.flatten()



# uncertain cart pole
ucp_error_data = np.load('examples/cart_pole/new cart pole/ucp_error_data.npz')
error_N2 = ucp_error_data['error_N2']
error_N3 = ucp_error_data['error_N3']
error_N4 = ucp_error_data['error_N4']
ax[0].semilogy(error_N2, linewidth=2, label='N=2')
ax[0].semilogy(error_N3, linewidth=2, label='N=3')
ax[0].semilogy(error_N4, linewidth=2, label='N=4')
ax[0].legend()
ax[0].set_xlabel('Inner-loop iteration')
ax[0].set_ylabel('Relative error')
ax[0].set_title('Uncertain Cart Pole')
ax[0].grid(alpha=0.2)

# CRM
crm_data = np.load('examples/consensus_aero_struct/consensus_aero_struct_albcd_solution.npz')
crm_error = crm_data['error']
ax[1].semilogy(crm_error, linewidth=2)
ax[1].set_xlabel('Inner-loop iteration')
ax[1].set_ylabel('Relative error')
ax[1].set_title('CRM')
ax[1].grid(alpha=0.2)

# MPCRM
mpcrm_data = np.load('examples/crm_aero_struct/multi_point_crm_error_data.npz')
error_N2 = mpcrm_data['error_N2']
error_N3 = mpcrm_data['error_N3']
error_N4 = mpcrm_data['error_N4']
ax[2].semilogy(error_N2, linewidth=2, label='N=2')
ax[2].semilogy(error_N3, linewidth=2, label='N=3')
ax[2].semilogy(error_N4, linewidth=2, label='N=4')
ax[2].legend()
ax[2].set_xlabel('Inner-loop iteration')
ax[2].set_ylabel('Relative error')
ax[2].set_title('Multi-Point CRM')
ax[2].grid(alpha=0.2)




plt.savefig('error_plot.pdf', bbox_inches='tight')
plt.show()