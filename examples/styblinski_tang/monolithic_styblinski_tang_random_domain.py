import numpy as np
import modopt as mo
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

# https://www.sfu.ca/~ssurjano/stybtang.html

dims = np.linspace(100, 1000, 10, dtype=int)
# dims = np.linspace(100, 300, 3, dtype=int)

jax_obj = lambda v: 0.5 * jnp.sum(v**4 - 16 * v**2 + 5 * v)

num = 30 # the number of random samples for each dimension
# num = 3 # the number of random samples for each dimension

mean, std = [], []

for n in dims:

    t = []

    for i in range(num):

        x0 = np.random.uniform(-5.0, 5.0, n)

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


file = 'monolithic_styblinski_tang_mean'
np.save(file, mean)

file = 'monolithic_styblinski_tang_std'
np.save(file, std)


plt.semilogy(dims, mean, 'o-', color='tab:orange')

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
    color="tab:orange",
    alpha=0.2,
)

plt.show()