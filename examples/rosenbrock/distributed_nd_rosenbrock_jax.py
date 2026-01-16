from optiplex import Plex
import jax.numpy as jnp
import modopt as mo
import numpy as np
import time

n = 1000 # dimension
N = 5 # number of subproblems

# n must be divisible by N
if n % N != 0: raise ValueError("n must be divisible by N")

objective, times = [], []

def make_sub_problem(subp, N, n):

    def sub_problem(x_init, y, mu):

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
        opt_time = t2 - t1
        times.append(opt_time + (times[-1] if len(times)>0 else 0))

        optimizer.print_results()

        objective.append(optimizer.results['fun'])

        x_init[subp] = optimizer.results['x']

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


opt = Plex(blocks=subP_functions,
           x_init=v_init)

opt.solve(max_iter=300,
          tol=1e-5)

print('Solution: ', opt.x_init)
print("Success:", opt.success)
print('Iterations: ', opt.num_iter)
# print('Time (s): ', opt.time)
print('Optimization time (s): ', times[-1])