import numpy as np
import matplotlib.pyplot as plt

path = 'examples/rosenbrock/data_ipopt_lenovo/'
# monolithic_mean = np.load(path + 'monolithic_rosenbrock_mean_n20_d2000_IPOPT.npy')
# monolithic_std = np.load(path + 'monolithic_rosenbrock_std_n20_d2000_IPOPT.npy')
monolithic_mean = np.load(path + 'monolithic_rosenbrock_mean_n20_d3000_IPOPT.npy')[::2]
monolithic_std = np.load(path + 'monolithic_rosenbrock_std_n20_d3000_IPOPT.npy')[::2]

# distributed_mean_2 = np.load(path + 'distributed_rosenbrock_mean_2subp_n20_d2000_IPOPT.npy')
# distributed_std_2 = np.load(path + 'distributed_rosenbrock_std_2subp_n20_d2000_IPOPT.npy')
distributed_mean_2 = np.load(path + 'distributed_rosenbrock_mean_2subp_n20_d3000_IPOPT.npy')[::2]
distributed_std_2 = np.load(path + 'distributed_rosenbrock_std_2subp_n20_d3000_IPOPT.npy')[::2]

# distributed_mean_5 = np.load(path + 'distributed_rosenbrock_mean_5subp_n20_d2000_IPOPT.npy')
# distributed_std_5 = np.load(path + 'distributed_rosenbrock_std_5subp_n20_d2000_IPOPT.npy')
distributed_mean_5 = np.load(path + 'distributed_rosenbrock_mean_5subp_n20_d3000_IPOPT.npy')[::2]
distributed_std_5 = np.load(path + 'distributed_rosenbrock_std_5subp_n20_d3000_IPOPT.npy')[::2]

# distributed_mean_10 = np.load(path + 'distributed_rosenbrock_mean_10subp_n20_d2000_IPOPT.npy')
# distributed_std_10 = np.load(path + 'distributed_rosenbrock_std_10subp_n20_d2000_IPOPT.npy')
distributed_mean_10 = np.load(path + 'distributed_rosenbrock_mean_10subp_n20_d3000_IPOPT.npy')[::2]
distributed_std_10 = np.load(path + 'distributed_rosenbrock_std_10subp_n20_d3000_IPOPT.npy')[::2]

# dims = np.linspace(100, 2000, 10, dtype=int)
dims = np.linspace(300, 3000, 5, dtype=int)



plt.figure(figsize=(4,3))

plt.semilogy(dims, monolithic_mean, marker='o', color='tab:gray', linewidth=1, markersize=7,label='Monolithic IPOPT', mec='k')

# plt.fill_between(
#     np.ravel(dims),
#     np.ravel(monolithic_mean - 2*monolithic_std),
#     np.ravel(monolithic_mean + 2*monolithic_std),
#     color="tab:blue",
#     alpha=0.2,
# )

plt.semilogy(dims, distributed_mean_2, marker='s', color='tab:blue', linewidth=1, markersize=7, label='Dist. (2 subproblems)', mec='k')

# plt.fill_between(
#     np.ravel(dims),
#     np.ravel(distributed_mean_2 - 2*distributed_std_2),
#     np.ravel(distributed_mean_2 + 2*distributed_std_2),
#     color="tab:blue",
#     alpha=0.2,
# )

plt.semilogy(dims, distributed_mean_5, marker='^', color='tab:orange', linewidth=1, markersize=7, label='Dist. (5 subproblems)', mec='k')

# plt.fill_between(
#     np.ravel(dims),
#     np.ravel(distributed_mean_5 - 2*distributed_std_5),
#     np.ravel(distributed_mean_5 + 2*distributed_std_5),
#     color="tab:orange",
#     alpha=0.2,
# )

plt.semilogy(dims, distributed_mean_10, marker='d', color='tab:green', linewidth=1, markersize=7, label='Dist. (10 subproblems)', mec='k')

# plt.fill_between(
#     np.ravel(dims),
#     np.ravel(distributed_mean_10 - 2*distributed_std_10),
#     np.ravel(distributed_mean_10 + 2*distributed_std_10),
#     color="tab:green",
#     alpha=0.2,
# )

plt.xlabel('Dimension (n)')
plt.ylabel('Optimization time (s)')
# plt.grid(color='lavender')
plt.legend(loc='lower right', fontsize=10)

# plt.xlim(min(dims), max(dims))

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_position(('outward', 6))
ax.spines['left'].set_position(('outward', 6))

ax.set_xticks([300, 3000])

plt.grid(color='blue', alpha=0.1, axis='y')

plt.savefig('rosenbrock_time_ipopt.pdf', bbox_inches='tight')

plt.show()