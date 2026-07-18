import numpy as np
import modopt as mo
import jax.numpy as jnp
import time
from scipy.stats.qmc import LatinHypercube, scale

n = 1000 # dimension

i = np.arange(1, n + 1, dtype=float)
solution_a = 2.0**(-(2.0**i - 2.0) / (2.0**i))

# solution_b = np.zeros(n)
# solution_b[0] = 1/3

num = 100
sampler = LatinHypercube(d=n, seed=42)
# samples = scale(sampler.random(num), l_bounds=-1 * np.ones(n), u_bounds=1 * np.ones(n))
samples = scale(sampler.random(num), l_bounds=solution_a - 0.2 * solution_a, u_bounds=solution_a + 0.2 * solution_a)

times = []
count = 0
for i in range(num):
    x0 = samples[i]

    jax_obj = lambda v: (v[0] - 1)**2 + jnp.sum(jnp.arange(2, n + 1) * (2 * v[1:]**2 - v[:-1])**2)

    jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 10000, 'ftol': 1e-7}, turn_off_outputs=True)

    t1 = time.perf_counter()

    res = optimizer.solve()

    t2 = time.perf_counter()

    # if res.success:
    #     times.append(t2 - t1)
    #     count += 1
    
    optimizer.print_results()
    ans = optimizer.results['x']
    # print('found solution: ', ans)
    # print(solution_b)

    error = np.inf
    for solution in [solution_a]:
        # error = min(error, np.linalg.norm((ans - solution) / solution))
        error = min(error, np.linalg.norm((ans - solution)))

    if res.success and error < 1e-2:
        times.append(t2 - t1)
        count += 1

    print('error: ', error)



mean_time = np.mean(times)
std_time = np.std(times)

print('Number of successful runs: ', count, ' out of ', num)

print('mean time (s): ', mean_time)
print('std time (s): ', std_time)



np.savez('dixon_price_monolithic_time_data_n1000_07172026.npz', times=times)