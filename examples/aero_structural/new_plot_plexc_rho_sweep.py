import numpy as np
import matplotlib.pyplot as plt

r11 = np.load('examples/aero_structural/history4_r1p1.npz')
time_r11 = r11['x_time']
error_r11 = r11['error']

r13 = np.load('examples/aero_structural/history4_r1p3.npz')
time_r13 = r13['x_time']
error_r13 = r13['error']

r15 = np.load('examples/aero_structural/history4_r1p5.npz')
time_r15 = r15['x_time']
error_r15 = r15['error']

plt.figure(figsize=(4, 3))
plt.semilogy(time_r11, error_r11, color='tab:blue', label=r'Error ($\rho=1.1$)', linewidth=2)
plt.semilogy(time_r13, error_r13, color='tab:orange', label=r'Error ($\rho=1.3$)', linewidth=2)
plt.semilogy(time_r15, error_r15, color='tab:green', label=r'Error ($\rho=1.5$)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.legend()
plt.grid(axis='y', color='lavender')
plt.savefig('simple_aero_struct_rho_sweep_2.pdf', bbox_inches='tight')
plt.show()