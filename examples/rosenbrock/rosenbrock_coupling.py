from optiplex import Plex
import jax.numpy as jnp
import modopt as mo
import numpy as np
import time
import tracemalloc
import gc
import warnings
warnings.filterwarnings("ignore")

n = 1000 # dimension
N = 5 # number of subproblems
beta = 10 # coupling strength

if n % N: raise ValueError("n must be divisible by N")

objective, times = [], []

def make_sub_problem(subp, N, n):

    def sub_problem(x_init, y, mu):

        def jax_obj(v):
            x_init[subp] = v

            x = jnp.concatenate(x_init)

            return jnp.sum(beta * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
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
# guess = np.array([-1.2, 1] * (n // 2))
v_init = []
for i in range(N): v_init.append(np.zeros(size))
# for i in range(N): v_init.append(guess[i*size:(i+1)*size])


opt = Plex(subproblems=subP_functions,
           x_init=v_init)

tracemalloc.start()

# opt.solve(max_iter=500, tol=1e-5, itol=1000,)
opt.solve(max_inner_iter=600,
          ATOL_in=1e-5, 
          RTOL_in=1e-5,
          )

current, peak = tracemalloc.get_traced_memory()
# print(f"Current: {current / 10**6:.2f} MB")
print(f"Peak: {peak / 10**6:.2f} MB")

tracemalloc.stop()

# print('Solution: ', opt.x)
# print('Total time (s): ', opt.time)
print('Optimization time (s): ', times[-1])


dimension = np.array([100, 500, 1000])

monolithic_time_slsqp_beta_10 = np.array([0.14, 6.25, 53.60])
monolithic_time_slsqp_beta_100 = np.array([0.40, 24.33, 200.43])
monolithic_time_slsqp_beta_200 = np.array([0.58, 31.71, 301.34])
monolithic_time_slsqp_beta_300 = np.array([0.52, 41.66, 344.51])


distributed_time_5_subp_slsqp_beta_10 = np.array([0.66, 1.36, 5.116])

distributed_time_5_subp_slsqp_beta_100 = np.array([4.62, 8.16, 24.61])

distributed_time_5_subp_slsqp_beta_200 = np.array([11.98, 15.76, 40.94])

distributed_time_5_subp_slsqp_beta_300 = np.array([13.95, 24.06, 56.19])