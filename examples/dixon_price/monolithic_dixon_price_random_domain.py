import numpy as np
import modopt as mo
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

# https://www.sfu.ca/~ssurjano/dixonpr.html

dims = np.linspace(100, 1000, 10, dtype=int)
# dims = np.linspace(100, 300, 3, dtype=int)

jax_obj = lambda v: (v[0] - 1)**2 + jnp.sum(jnp.arange(2, n + 1) * (2 * v[1:]**2 - v[:-1])**2)

num = 10 # the number of random samples for each dimension
# num = 3 # the number of random samples for each dimension

mean, std = [], []

for n in dims:

    t = []

    for i in range(num):

        # The function is usually evaluated on the hypercube xi ∈ [-10, 10], for all i = 1, …, d.
        x0 = np.random.uniform(-10.0, 10.0, n)

        jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 6000, 'ftol': 1e-8}, turn_off_outputs=True)

        t1 = time.perf_counter()

        optimizer.solve()

        t2 = time.perf_counter()

        optimizer.print_results()

        t.append(t2 - t1)

    mean.append(np.mean(t))
    std.append(np.std(t))




mean = np.array(mean)
std = np.array(std)


file = 'monolithic_dixon_price_mean'
np.save(file, mean)

file = 'monolithic_dixon_price_std'
np.save(file, std)


plt.semilogy(dims, mean, 'o-', color='tab:red')

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
    color="tab:red",
    alpha=0.2,
)

plt.show()