import numpy as np
import h5py
import matplotlib.pyplot as plt


path = 'examples/bwb/checkpoint copy.h5'

with h5py.File(path, 'r') as f:
	iteration_ids = sorted((int(key) for key in f.keys()))

	x_history = [f[str(iteration_id)]['x'][()] for iteration_id in iteration_ids]
	error_history = [f[str(iteration_id)]['error'][()] for iteration_id in iteration_ids]
	mu_history = [f[str(iteration_id)]['mu'][()] for iteration_id in iteration_ids]
	y_history = [f[str(iteration_id)]['y'][()] for iteration_id in iteration_ids]


n = iteration_ids[-1] + 1

plt.plot(iteration_ids, error_history, label='error')
plt.show()

# plot the convergence of all variables in the x vector
x_history = np.array(x_history)
for i in range(x_history.shape[1]):
    plt.plot(iteration_ids, x_history[:, i], label=f'x[{i}]')
	# plt.semilogy(iteration_ids, x_history[:, i] + 10, label=f'x[{i}]')
# plt.legend()
plt.show()