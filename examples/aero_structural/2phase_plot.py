import numpy as np
import matplotlib.pyplot as plt

data = np.load('examples/aero_structural/data/history4_r1p5.npz')

time = data['x_time']
error = data['error']
feasibility = data['feasibility']
mu = data['mu_history']

two_phase_data = np.load('examples/aero_structural/history4_2phase_r1p5_eps_1en1.npz')
# two_phase_data = np.load('examples/aero_structural/history4_2phase_r1p5.npz')
time_2phase = two_phase_data['x_time']
error_2phase = two_phase_data['error']
feasibility_2phase = two_phase_data['feasibility']
mu_2phase = two_phase_data['mu_history']

plt.figure(figsize=(4, 3))
plt.semilogy(time, error, color='tab:blue', linewidth=2, label='Exact')
plt.semilogy(time_2phase, error_2phase, color='tab:orange', label='2-phase adaptive', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.grid(axis='y', color='lavender')
plt.legend()
# plt.savefig('two_phase_aero_struct_error.png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()




plt.figure(figsize=(4, 3))
plt.semilogy(feasibility, color='tab:blue', label='Feasibility', linewidth=2)
plt.semilogy(feasibility_2phase, color='tab:orange', label='2-phase adaptive', linewidth=2, linestyle='--')
# add a dashed horizontal line at y=1e-3
plt.axhline(1e-3, color='gray', linestyle=':', label='_nolegend_')
# add text saying 'tolerance'
plt.text(5, 1e-3*1.5, 'Feasibility tolerance', fontsize=8, ha='center')
plt.xlabel('Outer-loop iteration')
plt.ylabel('Feasibility')
plt.grid(axis='y', color='lavender')
plt.legend()
# plt.savefig('two_phase_aero_struct_feasibility.png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()