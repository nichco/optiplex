import numpy as np
import matplotlib.pyplot as plt

data = np.load('examples/aero_structural/history4.npz')

time = data['x_time']
error = data['error']
# mu = data['mu_history']

plt.figure(figsize=(4, 3))
plt.semilogy(time, error, color='tab:blue', label='Error', linewidth=2)
plt.xlabel('Wall time (s)')
plt.ylabel('Error')
plt.grid(axis='y', color='lavender')

# plt.savefig('simple_aero_struct_error.pdf', bbox_inches='tight')
plt.show()