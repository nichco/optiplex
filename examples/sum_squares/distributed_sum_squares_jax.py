from optiplex import Plex
import jax.numpy as jnp
import modopt as mo
import numpy as np
import time
import tracemalloc
import gc
import warnings
warnings.filterwarnings("ignore")

n = 100 # dimension
N = 2 # number of subproblems

if n % N != 0: raise ValueError("n must be divisible by N")

objective, times = [], []

def make_sub_problem(subp, N, n):

    def sub_problem(x_init, y, mu):

        def jax_obj(v):
            x_init[subp] = v

            x = jnp.concatenate(x_init)

            return jnp.sum(jnp.arange(1, x.size + 1) * x**2)
        
        v0 = x_init[subp]
        
        jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf, order=1)

        optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 6000, 'ftol': 1e-7}, turn_off_outputs=True)
        # optimizer = mo.IPOPT(jaxprob, solver_options={'max_iter': 6000, 'tol': 1e-7}, turn_off_outputs=True)

        t1 = time.perf_counter()
        optimizer.solve()
        t2 = time.perf_counter()
        opt_time = t2 - t1
        times.append(opt_time + (times[-1] if len(times)>0 else 0))

        # optimizer.print_results()

        objective.append(optimizer.results['fun'])

        x_init[subp] = optimizer.results['x']

        gc.collect()

        return x_init
    
    return sub_problem




# run the function generator to generate subproblems
subP_functions = []
for i in range(N):
    subPfunc = make_sub_problem(i, N, n)
    subP_functions.append(subPfunc)

size = int(n / N)
v_init = []
for i in range(N): v_init.append(np.ones(size))


opt = Plex(blocks=subP_functions,
           x_init=v_init)

tracemalloc.start()

opt.solve(max_iter=500, tol=1e-5, itol=1e3,)

current, peak = tracemalloc.get_traced_memory()
# print(f"Current: {current / 10**6:.2f} MB")
print(f"Peak: {peak / 10**6:.2f} MB")

tracemalloc.stop()

# print('Solution: ', opt.x)
print("Success:", opt.success)
print('Iterations: ', opt.k)
# print('Time (s): ', opt.time)
print('Optimization time (s): ', times[-1])