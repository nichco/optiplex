import numpy as np
import matplotlib.pyplot as plt



dimension = np.array([100, 500, 1000])

monolithic_time_slsqp_beta_10 = np.array([0.14, 6.25, 53.60])
monolithic_time_slsqp_beta_100 = np.array([0.40, 24.33, 200.43])
monolithic_time_slsqp_beta_200 = np.array([0.58, 31.71, 301.34])
monolithic_time_slsqp_beta_300 = np.array([0.52, 41.66, 344.51])

distributed_time_5_subp_slsqp_beta_10 = np.array([0.66, 1.36, 5.116])

distributed_time_5_subp_slsqp_beta_100 = np.array([4.62, 8.16, 24.61])

distributed_time_5_subp_slsqp_beta_200 = np.array([11.98, 15.76, 40.94])

distributed_time_5_subp_slsqp_beta_300 = np.array([13.95, 24.06, 56.19])


delta_b10 = distributed_time_5_subp_slsqp_beta_10 / monolithic_time_slsqp_beta_10
delta_b100 = distributed_time_5_subp_slsqp_beta_100 / monolithic_time_slsqp_beta_100
delta_b200 = distributed_time_5_subp_slsqp_beta_200 / monolithic_time_slsqp_beta_200
delta_b300 = distributed_time_5_subp_slsqp_beta_300 / monolithic_time_slsqp_beta_300


plt.figure(figsize=(4,3))

plt.plot(dimension, delta_b10, marker='o', label=r'$\beta=10$', color='tab:red', linewidth=1, markersize=7)
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

plt.text(1.0, 0.45, 'Monolithic is faster', transform=plt.gca().transAxes, ha='right', va='center', fontsize=8, color='gray')
plt.text(1.0, 0.39, 'Distributed is faster', transform=plt.gca().transAxes, ha='right', va='center', fontsize=8, color='gray')

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