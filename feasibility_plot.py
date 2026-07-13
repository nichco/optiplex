import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = plt.rcParams['axes.labelsize']

fig, ax = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)
ax = ax.flatten()



# uncertain cart pole
ucp_data_N2 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N2_V2.npz')
feas_N2 = ucp_data_N2['feasibility'][1:]
ucp_data_N3 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N3_V2.npz')
feas_N3 = ucp_data_N3['feasibility'][1:]
ucp_data_N4 = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_distributed_history_N4_V2.npz')
feas_N4 = ucp_data_N4['feasibility'][1:]
ax[0].semilogy(feas_N2, linewidth=2, label='N=2')
ax[0].semilogy(feas_N3, linewidth=2, label='N=3')
ax[0].semilogy(feas_N4, linewidth=2, label='N=4')
ax[0].legend()
ax[0].set_xlabel('Outer-loop iteration')
ax[0].set_ylabel('Feasibility: $\\Vert c \\Vert_\infty$')
ax[0].set_title('Uncertain Cart Pole')
ax[0].grid(alpha=0.2)

# CRM
crm_data = np.load('examples/consensus_aero_struct/consensus_aero_struct_albcd_solution.npz')
crm_feas = crm_data['feasibility'][1:]
ax[1].semilogy(crm_feas, linewidth=2)
ax[1].set_xlabel('Outer-loop iteration')
ax[1].set_ylabel('Feasibility: $\\Vert c \\Vert_\infty$')
ax[1].set_title('CRM')
ax[1].grid(alpha=0.2)

# # MPCRM
mpcrm_data_N2 = np.load('examples/crm_aero_struct/distributed_solution_N2_V2.npz')
feas_N2 = mpcrm_data_N2['feasibility'][1:]
mpcrm_data_N3 = np.load('examples/crm_aero_struct/distributed_solution_N3_V2.npz')
feas_N3 = mpcrm_data_N3['feasibility'][1:]
mpcrm_data_N4 = np.load('examples/crm_aero_struct/distributed_solution_N4_V2.npz')
feas_N4 = mpcrm_data_N4['feasibility'][1:]
ax[2].semilogy(feas_N2, linewidth=2, label='N=2')
ax[2].semilogy(feas_N3, linewidth=2, label='N=3')
ax[2].semilogy(feas_N4, linewidth=2, label='N=4')
ax[2].legend()
ax[2].set_xlabel('Outer-loop iteration')
ax[2].set_ylabel('Feasibility: $\\Vert c \\Vert_\infty$')
ax[2].set_title('Multi-Point CRM')
ax[2].grid(alpha=0.2)




plt.savefig('feas_plot.pdf', bbox_inches='tight')
plt.show()