import numpy as np
import matplotlib.pyplot as plt





monolithic_mean_time = np.load('monolithic_rosenbrock_mean.npy')
monolithic_std_time = np.load('monolithic_rosenbrock_std.npy')




dims = np.linspace(100, 1000, 10, dtype=int)



plt.semilogy(dims, monolithic_mean_time, 'o-', color='tab:blue', label='Monolithic SLSQP')

plt.fill_between(
    np.ravel(dims),
    np.ravel(monolithic_mean_time - monolithic_std_time),
    np.ravel(monolithic_mean_time + monolithic_std_time),
    color="tab:blue",
    alpha=0.2,
)

plt.show()