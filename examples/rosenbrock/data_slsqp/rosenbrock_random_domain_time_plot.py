import numpy as np
import matplotlib.pyplot as plt

path = 'examples/rosenbrock/data/'
monolithic_mean = np.load(path + 'monolithic_rosenbrock_mean_n20.npy')
monolithic_std = np.load(path + 'monolithic_rosenbrock_std_n20.npy')

distributed_mean_2 = np.load(path + 'distributed_rosenbrock_mean_2subp.npy')
distributed_std_2 = np.load(path + 'distributed_rosenbrock_std_2subp.npy')

distributed_mean_5 = np.load(path + 'distributed_rosenbrock_mean_5subp.npy')
distributed_std_5 = np.load(path + 'distributed_rosenbrock_std_5subp.npy')

distributed_mean_10 = np.load(path + 'distributed_rosenbrock_mean_10subp.npy')
distributed_std_10 = np.load(path + 'distributed_rosenbrock_std_10subp.npy')

dims = np.linspace(100, 1000, 10, dtype=int)



plt.figure(figsize=(4,3))

plt.semilogy(dims, monolithic_mean, 'o-', color='tab:blue', linewidth=2, markersize=8,label='Monolithic SLSQP')

plt.fill_between(
    np.ravel(dims),
    np.ravel(monolithic_mean - 2*monolithic_std),
    np.ravel(monolithic_mean + 2*monolithic_std),
    color="tab:blue",
    alpha=0.2,
)

plt.semilogy(dims, distributed_mean_2, 's-', color='tab:orange', linewidth=2, markersize=8, label='Dist. SLSQP (2 subproblems)')

plt.fill_between(
    np.ravel(dims),
    np.ravel(distributed_mean_2 - 2*distributed_std_2),
    np.ravel(distributed_mean_2 + 2*distributed_std_2),
    color="tab:orange",
    alpha=0.2,
)

plt.semilogy(dims, distributed_mean_5, '^-', color='tab:green', linewidth=2, markersize=8, label='Dist. SLSQP (5 subproblems)')

plt.fill_between(
    np.ravel(dims),
    np.ravel(distributed_mean_5 - 2*distributed_std_5),
    np.ravel(distributed_mean_5 + 2*distributed_std_5),
    color="tab:green",
    alpha=0.2,
)

plt.semilogy(dims, distributed_mean_10, 'v-', color='tab:red', linewidth=2, markersize=8, label='Dist. SLSQP (10 subproblems)')

plt.fill_between(
    np.ravel(dims),
    np.ravel(distributed_mean_10 - 2*distributed_std_10),
    np.ravel(distributed_mean_10 + 2*distributed_std_10),
    color="tab:red",
    alpha=0.2,
)

plt.xlabel('Dimension (n)')
plt.ylabel('Optimization time (s)')
plt.grid(color='lavender')
plt.legend(loc='lower right', fontsize=10)

plt.xlim(min(dims), max(dims))

plt.savefig('rosenbrock_random_time.pdf', bbox_inches='tight')

plt.show()