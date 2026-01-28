import numpy as np
from optiplex import Plex
import jax.numpy as jnp
import modopt as mo
import time
import matplotlib.pyplot as plt
import gc
import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

# https://www.sfu.ca/~ssurjano/rosen.html

tvar = 0 # stores the optimization time for each subproblem

def make_sub_problem(subp, N, n):

    def sub_problem(x_init, y, mu):

        global tvar # basically magic

        def jax_obj(v):
            x_init[subp] = v
            x = jnp.concatenate(x_init)
            return jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        v0 = x_init[subp]
        
        jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf, order=1)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 6000, 'ftol': 1e-7}, turn_off_outputs=True)
        # optimizer = mo.IPOPT(jaxprob, solver_options={'max_iter': 6000, 'tol': 1e-7}, turn_off_outputs=True)

        t1 = time.perf_counter()
        optimizer.solve()
        t2 = time.perf_counter()
        tvar += t2 - t1

        # optimizer.print_results()

        x_init[subp] = optimizer.results['x']

        gc.collect()

        return x_init
    
    return sub_problem













dims = np.linspace(100, 1000, 10, dtype=int)
# dims = np.linspace(100, 300, 3, dtype=int)

N = 2 # number of subproblems

num = 20 # the number of random samples for each dimension
# num = 3 # the number of random samples for each dimension

mean, std = [], []

for n in dims:

    if n % N: raise ValueError("n must be divisible by N")

    t = []

    # run the function generator to generate subproblems
    subP_functions = []
    for i in range(N):
        subPfunc = make_sub_problem(i, N, n)
        subP_functions.append(subPfunc)

    for i in range(num):

        tvar = 0 # reset tvar for each run

        v_init = np.random.uniform(-1.0, 1.0, n)

        size = int(n / N)
        v_init = [v_init[i*size:(i+1)*size] for i in range(N)]


        opt = Plex(blocks=subP_functions, x_init=v_init)

        opt.solve(max_iter=500, tol=1e-5, itol=1e3,)

        print("Success:", opt.success)
        print('Iterations: ', opt.dual_iterations)
        print('Optimization time (s): ', tvar)

        t.append(tvar)

    mean.append(np.mean(t))
    std.append(np.std(t))




mean = np.array(mean)
std = np.array(std)


file = 'distributed_rosenbrock_mean_2subp'
np.save(file, mean)

file = 'distributed_rosenbrock_std_2subp'
np.save(file, std)


plt.semilogy(dims, mean, 'o-', color='tab:blue')

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
    color="tab:blue",
    alpha=0.2,
)

plt.show()