import numpy as np
import modopt as mo
import jax.numpy as jnp
import time
from scipy.stats.qmc import LatinHypercube, scale

# https://www.sfu.ca/~ssurjano/rosen.html

n = 1000 # dimension

num = 100
sampler = LatinHypercube(d=n, seed=42)
samples = scale(sampler.random(num), l_bounds=-1.5 * np.ones(n), u_bounds=1.5 * np.ones(n))

times = []
count = 0
for i in range(num):
    x0 = samples[i]

    jax_obj = lambda v: jnp.sum(100 * (v[1:] - v[:-1]**2)**2 + (1 - v[:-1])**2)

    jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 10000, 'ftol': 1e-7}, turn_off_outputs=True)

    t1 = time.perf_counter()

    res = optimizer.solve()

    t2 = time.perf_counter()

    if res.success:
        times.append(t2 - t1)
        count += 1
    
    optimizer.print_results()
    ans = optimizer.results['x']



mean_time = np.mean(times)
std_time = np.std(times)

print('Number of successful runs: ', count, ' out of ', num)

print('mean time (s): ', mean_time)
print('std time (s): ', std_time)



np.savez('monolithic_time_data_n1000_07162026.npz', times=times)