from joblib import Parallel, delayed
import numpy as np
import modopt as mo
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

# https://www.sfu.ca/~ssurjano/dixonpr.html

dims = np.linspace(100, 2000, 10, dtype=int)

num = 20 # the number of random samples for each dimension

jax_obj = lambda v: jnp.sum(100 * (v[1:] - v[:-1]**2)**2 + (1 - v[:-1])**2)

def run_dims(n):

    t = []

    for _ in range(num):

        x0 = np.random.uniform(-1.0, 1.0, n)

        jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

        # optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 6000, 'ftol': 1e-8}, turn_off_outputs=True)
        optimizer = mo.IPOPT(jaxprob, solver_options={'max_iter': 20000, 'tol': 1e-8}, turn_off_outputs=True)

        t1 = time.perf_counter()

        optimizer.solve()

        t2 = time.perf_counter()

        optimizer.print_results()

        t.append(t2 - t1)

    return np.mean(t), np.std(t)


results = Parallel(n_jobs=-1, backend="loky")(delayed(run_dims)(n) for n in dims)

mean, std = map(np.array, zip(*results))


mean = np.array(mean)
std = np.array(std)


file = 'monolithic_rosenbrock_mean_n20_d2000_IPOPT.npy'
np.save(file, mean)

file = 'monolithic_rosenbrock_std_n20_d2000_IPOPT.npy'
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