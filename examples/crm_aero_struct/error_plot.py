import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(3, 3))

n_twist_cp = 17
n_thickness_cp = 10

N = 2
solution_N2 = np.load('examples/crm_aero_struct/test_solution_d.npz')
alphas_star = np.array(solution_N2['alphas']).reshape(-1, 1)
twist_cp_star = solution_N2['twist_cp']
thickness_cp_star = solution_N2['thickness_cp']

twist_params = np.broadcast_to(twist_cp_star, (N, n_twist_cp))
thickness_params = np.broadcast_to(thickness_cp_star, (N, n_thickness_cp))
solution_N2 = np.concatenate((alphas_star, twist_params, thickness_params), axis=1).ravel()

data_N2 = np.load('examples/crm_aero_struct/distributed_solution_N2.npz')
h2 = np.array(data_N2['history'])

error_N2 = np.zeros(len(h2))
for i in range(len(h2)):
    h_i = h2[i].flatten()
    error_i = np.linalg.norm((h_i - solution_N2))
    error_N2[i] = error_i

plt.semilogy(error_N2, label='N=2', linewidth=2)





N = 3
solution_N3 = np.load('examples/crm_aero_struct/test_solution_N3.npz')
alphas_star = np.array(solution_N3['alphas']).reshape(-1, 1)
twist_cp_star = solution_N3['twist_cp']
thickness_cp_star = solution_N3['thickness_cp']

twist_params = np.broadcast_to(twist_cp_star, (N, n_twist_cp))
thickness_params = np.broadcast_to(thickness_cp_star, (N, n_thickness_cp))
solution_N3 = np.concatenate((alphas_star, twist_params, thickness_params), axis=1).ravel()

data_N3 = np.load('examples/crm_aero_struct/distributed_solution_N3_V2.npz')
h3 = np.array(data_N3['history'])

error_N3 = np.zeros(len(h3))
for i in range(len(h3)):
    h_i = h3[i].flatten()
    error_i = np.linalg.norm((h_i - solution_N3))
    error_N3[i] = error_i

plt.semilogy(error_N3, label='N=3', linewidth=2)





N = 4
solution_N4 = np.load('examples/crm_aero_struct/test_solution_N4.npz')
alphas_star = np.array(solution_N4['alphas']).reshape(-1, 1)
twist_cp_star = solution_N4['twist_cp']
thickness_cp_star = solution_N4['thickness_cp']

twist_params = np.broadcast_to(twist_cp_star, (N, n_twist_cp))
thickness_params = np.broadcast_to(thickness_cp_star, (N, n_thickness_cp))
solution_N4 = np.concatenate((alphas_star, twist_params, thickness_params), axis=1).ravel()

data_N4 = np.load('examples/crm_aero_struct/distributed_solution_N4.npz')
h4 = np.array(data_N4['history'])

error_N4 = np.zeros(len(h4))
for i in range(len(h4)):
    h_i = h4[i].flatten()
    error_i = np.linalg.norm((h_i - solution_N4))
    error_N4[i] = error_i

plt.semilogy(error_N4, label='N=4', linewidth=2)









plt.title('Multi-Point CRM')
plt.ylabel('Error')
plt.xlabel('Iteration')
plt.grid(alpha=0.2)
plt.legend()

# plt.savefig('examples/crm_aero_struct/error_plot.pdf', bbox_inches='tight')
plt.show()



np.savez('examples/crm_aero_struct/multi_point_crm_error_data.npz',
         error_N2=error_N2,
         error_N3=error_N3,
         error_N4=error_N4,
)