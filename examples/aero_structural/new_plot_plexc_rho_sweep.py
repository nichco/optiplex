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

r20 = np.load('examples/aero_structural/history4_r2p0.npz')
time_r20 = r20['x_time']
error_r20 = r20['error']

r25 = np.load('examples/aero_structural/history4_r2p5.npz')
time_r25 = r25['x_time']
error_r25 = r25['error']

r30 = np.load('examples/aero_structural/history4_r3p0.npz')
time_r30 = r30['x_time']
error_r30 = r30['error']

plt.figure(figsize=(4, 3))
plt.semilogy(time_r11, error_r11, color='tab:blue', label=r'Error ($\rho=1.1$)', linewidth=2)
plt.semilogy(time_r13, error_r13, color='tab:orange', label=r'Error ($\rho=1.3$)', linewidth=2)
plt.semilogy(time_r15, error_r15, color='tab:green', label=r'Error ($\rho=1.5$)', linewidth=2)
plt.semilogy(time_r20, error_r20, color='tab:red', label=r'Error ($\rho=2.0$)', linewidth=2)
plt.semilogy(time_r25, error_r25, color='tab:purple', label=r'Error ($\rho=2.5$)', linewidth=2)
plt.semilogy(time_r30, error_r30, color='tab:brown', label=r'Error ($\rho=3.0$)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.legend()
plt.grid(axis='y', color='lavender')
# plt.savefig('simple_aero_struct_rho_sweep_2.pdf', bbox_inches='tight')
plt.show()