import numpy as np
import matplotlib.pyplot as plt

data = np.load('examples/aero_structural/history_idf.npz')

time = data['x_time']
error = data['error']

data_pen_1e2 = np.load('examples/aero_structural/history_pen_1e2.npz')
error_pen_1e2 = data_pen_1e2['error']
time_pen_1e2 = data_pen_1e2['x_time']

data_pen_5e2 = np.load('examples/aero_structural/history_pen_5e2.npz')
error_pen_5e2 = data_pen_5e2['error']
time_pen_5e2 = data_pen_5e2['x_time']

data_pen_9e2 = np.load('examples/aero_structural/history_pen_9e2.npz')
error_pen_9e2 = data_pen_9e2['error']
time_pen_9e2 = data_pen_9e2['x_time']

data_pen_1e3 = np.load('examples/aero_structural/history_pen_1e3.npz')
error_pen_1e3 = data_pen_1e3['error']
time_pen_1e3 = data_pen_1e3['x_time']

data_pen_2e3 = np.load('examples/aero_structural/history_pen_2e3.npz')
error_pen_2e3 = data_pen_2e3['error']
time_pen_2e3 = data_pen_2e3['x_time']

data_pen_3e3 = np.load('examples/aero_structural/history_pen_3e3.npz')
error_pen_3e3 = data_pen_3e3['error']
time_pen_3e3 = data_pen_3e3['x_time']

data_pen_4e3 = np.load('examples/aero_structural/history_pen_4e3.npz')
error_pen_4e3 = data_pen_4e3['error']
time_pen_4e3 = data_pen_4e3['x_time']

data_pen_5e3 = np.load('examples/aero_structural/history_pen_5e3.npz')
error_pen_5e3 = data_pen_5e3['error']
time_pen_5e3 = data_pen_5e3['x_time']

data_pen_1e4 = np.load('examples/aero_structural/history_pen_1e4.npz')
error_pen_1e4 = data_pen_1e4['error']
time_pen_1e4 = data_pen_1e4['x_time']


plt.figure(figsize=(4,3))
plt.grid(axis='y', color='lavender', zorder=-10)

# plt.semilogy(time, error, label='AL-BCD', linewidth=1.5, marker='o', markersize=4, markevery=12)
plt.semilogy(time, error, label='AL-BCD', linewidth=2)

# plt.semilogy(time_pen_1e2, error_pen_1e2, label='Penalty (1e2)', linewidth=1.5)

# plt.semilogy(time_pen_5e2, error_pen_5e2, label='Penalty (5e2)', linewidth=1.5)

# plt.semilogy(time_pen_9e2, error_pen_9e2, label='Penalty (9e2)', linewidth=1.5, marker='v', markersize=4, markevery=30)
plt.semilogy(time_pen_9e2, error_pen_9e2, label='Penalty (9e2)', linewidth=2)

# plt.semilogy(time_pen_1e3, error_pen_1e3, label='Penalty (1e3)', linewidth=1.5, marker='s', markersize=4, markevery=30)
plt.semilogy(time_pen_1e3, error_pen_1e3, label='Penalty (1e3)', linewidth=2)

# plt.semilogy(time_pen_2e3, error_pen_2e3, label='Penalty (2e3)', linewidth=1.5, marker='^', markersize=4, markevery=30)
plt.semilogy(time_pen_2e3, error_pen_2e3, label='Penalty (2e3)', linewidth=2)

# plt.semilogy(time_pen_3e3, error_pen_3e3, label='Penalty (3e3)', linewidth=1.5, marker='d', markersize=4, markevery=30)
plt.semilogy(time_pen_3e3, error_pen_3e3, label='Penalty (3e3)', linewidth=2)

# plt.semilogy(time_pen_4e3, error_pen_4e3, label='Penalty (4e3)', linewidth=1.5, marker='*', markersize=4, markevery=30)
plt.semilogy(time_pen_4e3, error_pen_4e3, label='Penalty (4e3)', linewidth=2)

# plt.semilogy(time_pen_5e3, error_pen_5e3, label='Penalty (5e3)', linewidth=1.5)
# plt.semilogy(time_pen_5e3, error_pen_5e3, label='Penalty (5e3)', linewidth=2)

# plt.semilogy(time_pen_1e4, error_pen_1e4, label='Penalty (1e4)', linewidth=1.5)


plt.xlabel('Wall time (s)')
plt.ylabel('Error')
# plt.legend(loc='lower right', fontsize='small')
plt.legend(loc='lower right')
# plt.grid(axis='y', color='lavender', zorder=0)

plt.savefig('simple_error_comp.pdf', bbox_inches='tight')
plt.show()