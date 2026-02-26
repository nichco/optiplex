import numpy as np
import matplotlib.pyplot as plt



dimension = np.array([100, 500, 1000])

monolithic_time_slsqp_beta_100 = np.array([0.40, 24.33, 200.43])
monolithic_time_slsqp_beta_200 = np.array([0.58, 31.71, 301.34])
monolithic_time_slsqp_beta_300 = np.array([0.52, 41.66, 344.51])

distributed_time_5_subp_slsqp_beta_100 = np.array([4.62, 8.16, 24.61])

distributed_time_5_subp_slsqp_beta_200 = np.array([11.98, 15.76, 40.94])

distributed_time_5_subp_slsqp_beta_300 = np.array([13.95, 24.06, 56.19])



delta_b100 = distributed_time_5_subp_slsqp_beta_100 / monolithic_time_slsqp_beta_100
delta_b200 = distributed_time_5_subp_slsqp_beta_200 / monolithic_time_slsqp_beta_200
delta_b300 = distributed_time_5_subp_slsqp_beta_300 / monolithic_time_slsqp_beta_300


plt.figure(figsize=(4,3))

plt.semilogy(dimension, delta_b100, marker='o', label=r'$\beta=100$', color='tab:blue', linewidth=1, markersize=7)
plt.semilogy(dimension, delta_b200, marker='s', label=r'$\beta=200$', color='tab:orange', linewidth=1, markersize=7)
plt.semilogy(dimension, delta_b300, marker='d', label=r'$\beta=300$', color='tab:green', linewidth=1, markersize=7)

# add horizontal line at y=1
plt.axhline(y=1, color='tab:gray', linestyle='--', linewidth=1)

# plt.semilogy(dimension, monolithic_time_slsqp_beta_100, marker='o', label='Monolithic SLSQP (beta=100)', color='tab:gray', linewidth=1, markersize=7)
# plt.semilogy(dimension, monolithic_time_slsqp_beta_200, marker='o', label='Monolithic SLSQP (beta=200)', color='tab:gray', linewidth=1, markersize=7)
# plt.semilogy(dimension, monolithic_time_slsqp_beta_300, marker='o', label='Monolithic SLSQP (beta=300)', color='tab:gray', linewidth=1, markersize=7)
# plt.semilogy(dimension, distributed_time_5_subp_slsqp_beta_100, marker='s', label='Dist. (5 subproblems, beta=100)', color='tab:orange', linewidth=1, markersize=7)
# plt.semilogy(dimension, distributed_time_5_subp_slsqp_beta_200, marker='^', label='Dist. (5 subproblems, beta=200)', color='tab:blue', linewidth=1, markersize=7)
# plt.semilogy(dimension, distributed_time_5_subp_slsqp_beta_300, marker='d', label='Dist. (5 subproblems, beta=300)', color='tab:green', linewidth=1, markersize=7)


plt.xlabel('Dimension (n)')
plt.ylabel('Distributed Time / Monolithic Time')
plt.legend(loc='upper right', fontsize=10)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_position(('outward', 6))
ax.spines['left'].set_position(('outward', 6))

ax.set_xticks([100, 1000])

plt.savefig('rosenbrock_coupling.pdf', bbox_inches='tight')
plt.show()


"""
coupling = np.array([100, 200, 300])

monolithic_time_slsqp_n100 = np.array([0.40, 0.58, 0.52])
monolithic_time_slsqp_n500 = np.array([24.33, 31.71, 41.66])
monolithic_time_slsqp_n1000 = np.array([200.43, 301.34, 344.51])

distributed_time_5_subp_slsqp_n100 = np.array([4.62, 11.98, 13.95])
distributed_time_5_subp_slsqp_n500 = np.array([8.16, 15.76, 24.06])
distributed_time_5_subp_slsqp_n1000 = np.array([24.61, 40.94, 56.19])

delta_n100 = distributed_time_5_subp_slsqp_n100 / monolithic_time_slsqp_n100
delta_n500 = distributed_time_5_subp_slsqp_n500 / monolithic_time_slsqp_n500
delta_n1000 = distributed_time_5_subp_slsqp_n1000 / monolithic_time_slsqp_n1000

plt.plot(coupling, delta_n100, marker='o', label='Time ratio (Dist. / Monolithic) (n=100)', color='tab:purple', linewidth=1, markersize=7)
plt.plot(coupling, delta_n500, marker='s', label='Time ratio (Dist. / Monolithic) (n=500)', color='tab:orange', linewidth=1, markersize=7)
plt.plot(coupling, delta_n1000, marker='d', label='Time ratio (Dist. / Monolithic) (n=1000)', color='tab:green', linewidth=1, markersize=7)

# plt.figure(figsize=(4,3))

# plt.semilogy(coupling, monolithic_time_slsqp_n100, marker='o', label='Monolithic SLSQP (n=100)', color='tab:gray', linewidth=1, markersize=7)
# plt.semilogy(coupling, monolithic_time_slsqp_n500, marker='o', label='Monolithic SLSQP (n=500)', color='tab:gray', linewidth=1, markersize=7)
# plt.semilogy(coupling, monolithic_time_slsqp_n1000, marker='o', label='Monolithic SLSQP (n=1000)', color='tab:gray', linewidth=1, markersize=7)
# plt.semilogy(coupling, distributed_time_5_subp_slsqp_n100, marker='s', label='Dist. (5 subproblems, n=100)', color='tab:orange', linewidth=1, markersize=7)
# plt.semilogy(coupling, distributed_time_5_subp_slsqp_n500, marker='^', label='Dist. (5 subproblems, n=500)', color='tab:blue', linewidth=1, markersize=7)
# plt.semilogy(coupling, distributed_time_5_subp_slsqp_n1000, marker='d', label='Dist. (5 subproblems, n=1000)', color='tab:green', linewidth=1, markersize=7)


plt.xlabel('Coupling strength (beta)')
plt.ylabel('Optimization time (s)')
# plt.legend(loc='lower right', fontsize=10)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_position(('outward', 6))
ax.spines['left'].set_position(('outward', 6))

plt.show()
"""