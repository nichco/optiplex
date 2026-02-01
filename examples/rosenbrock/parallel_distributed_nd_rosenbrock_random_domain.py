from joblib import Parallel, delayed
from optiplex import Plex
import numpy as np
import modopt as mo
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import warnings
import gc
import time


def make_sub_problem(subp, N, n, timer):

    def sub_problem(x_init, y, mu):

        def jax_obj(v):
            x_init[subp] = v

            x = jnp.concatenate(x_init)

            return jnp.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        v0 = x_init[subp]
        
        jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf, order=1)

        # optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 6000, 'ftol': 1e-7}, turn_off_outputs=True)
        optimizer = mo.IPOPT(jaxprob, solver_options={'max_iter': 20000, 'tol': 1e-7}, turn_off_outputs=True)

        t1 = time.perf_counter()
        optimizer.solve()
        t2 = time.perf_counter()
        timer[subp] += t2 - t1

        # optimizer.print_results()

        x_init[subp] = optimizer.results['x']

        gc.collect()

        return x_init
    
    return sub_problem




np.random.seed(0)

dims = np.linspace(100, 2000, 10, dtype=int)
# dims = np.linspace(100, 400, 4, dtype=int)

num = 20 # the number of random samples for each dimension
# num = 2 # the number of random samples for each dimension

N = 10 # number of subproblems

jax_obj = lambda v: jnp.sum(100 * (v[1:] - v[:-1]**2)**2 + (1 - v[:-1])**2)

def run_dims(n):

    warnings.filterwarnings("ignore")

    t = []

    size = int(n / N)

    for _ in range(num):

        timer = np.zeros(N)

        # run the function generator to generate subproblems
        subP_functions = []
        for i in range(N):
            subPfunc = make_sub_problem(i, N, n, timer)
            subP_functions.append(subPfunc)



        x0 = np.random.uniform(-1.0, 1.0, n)
        v_init = [x0[i*size:(i+1)*size] for i in range(N)]

        opt = Plex(blocks=subP_functions,
           x_init=v_init)
        
        opt.solve(max_iter=500, tol=1e-5, itol=1e3,)

        print("Success:", opt.success)
        print('Iterations: ', opt.dual_iterations)

        t.append(np.sum(timer))

        

    return np.mean(t), np.std(t)






results = Parallel(n_jobs=-1, backend="loky")(delayed(run_dims)(n) for n in dims)

mean, std = map(np.array, zip(*results))


mean = np.array(mean)
std = np.array(std)


file = 'distributed_rosenbrock_mean_10subp_n20_d2000_IPOPT'
np.save(file, mean)

file = 'distributed_rosenbrock_std_10subp_n20_d2000_IPOPT'
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