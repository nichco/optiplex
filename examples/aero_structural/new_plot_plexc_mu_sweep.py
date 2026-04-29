import numpy as np
import matplotlib.pyplot as plt

m1 = np.load('examples/aero_structural/history4_r1p5_m1.npz')
time_m1 = m1['x_time']
error_m1 = m1['error']

m10 = np.load('examples/aero_structural/history4_r1p5_m10.npz')
time_m10 = m10['x_time']
error_m10 = m10['error']

m100 = np.load('examples/aero_structural/history4_r1p5_m100.npz')
time_m100 = m100['x_time']
error_m100 = m100['error']

m1000 = np.load('examples/aero_structural/history4_r1p5_m1000.npz')
time_m1000 = m1000['x_time']
error_m1000 = m1000['error']

plt.figure(figsize=(4, 3))
plt.semilogy(time_m1, error_m1, color='tab:blue', label=r'$\mu^0=1$', linewidth=2)
plt.semilogy(time_m10, error_m10, color='tab:orange', label=r'$\mu^0=10$', linewidth=2)
plt.semilogy(time_m100, error_m100, color='tab:green', label=r'$\mu^0=100$', linewidth=2)
plt.semilogy(time_m1000, error_m1000, color='tab:red', label=r'$\mu^0=1000$', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.legend()
plt.grid(axis='y', color='lavender')
plt.savefig('simple_aero_struct_mu_sweep_2.pdf', bbox_inches='tight')
plt.show()