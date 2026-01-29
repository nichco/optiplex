import numpy as np
import modopt as mo
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

# https://www.sfu.ca/~ssurjano/zakharov.html

dims = np.linspace(100, 1000, 10, dtype=int)
# dims = np.linspace(100, 500, 5, dtype=int)

# # original Zakharov function (blows up at high dimensions)
# def jax_obj(v):
#     s = 0.5 * jnp.sum(jnp.arange(1, n + 1) * v)
#     return jnp.sum(v**2) + s**2 + s**4

# modified Zakharov function (normalized by dimension)
def jax_obj(v):
    s = 0.5 * jnp.sum(jnp.arange(1, n + 1) * v) / n
    return jnp.sum(v**2) + s**2 + s**4

# num = 20 # the number of random samples for each dimension
num = 10 # the number of random samples for each dimension

mean, std = [], []

for n in dims:

    t = []

    for i in range(num):

        x0 = np.random.uniform(-5.0, 5.0, n)

        jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-7}, turn_off_outputs=True)

        t1 = time.perf_counter()

        optimizer.solve()

        t2 = time.perf_counter()

        optimizer.print_results()

        t.append(t2 - t1)

    mean.append(np.mean(t))
    std.append(np.std(t))




mean = np.array(mean)
std = np.array(std)


# file = 'monolithic_zakharov_mean_n3'
# np.save(file, mean)

# file = 'monolithic_zakharov_std_n3'
# np.save(file, std)


plt.semilogy(dims, mean, 'o-', color='tab:green')

# plt.fill_between(
#     np.ravel(dims),
#     np.ravel(mean - 3 * np.sqrt(std)),
#     np.ravel(mean + 3 * np.sqrt(std)),
#     color="lightgrey",
# )
plt.fill_between(
    np.ravel(dims),
    np.ravel(mean - std),
    np.ravel(mean + std),
    color="tab:green",
    alpha=0.2,
)

plt.xlabel('Dimension')
plt.ylabel('Time (s)')

plt.show()