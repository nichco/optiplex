import numpy as np
import modopt as mo
import jax.numpy as jnp
import tracemalloc

# https://www.sfu.ca/~ssurjano/rosen.html

n = 1000 # dimension
beta = 100 # coupling strength

tracemalloc.start()

jax_obj = lambda v: jnp.sum(beta * (v[1:] - v[:-1]**2)**2 + (1 - v[:-1])**2)
    
# guess = np.array([-1.2, 1] * (n // 2))
# x0 = guess
x0 = np.zeros((n,))
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 5000, 'ftol': 1e-7}, turn_off_outputs=True)
# optimizer = mo.InteriorPoint(jaxprob, recording=False, turn_off_outputs=True, maxiter=6000, opt_tol=1e-7, feas_tol=1e-7)

# tracemalloc.start()

optimizer.solve()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

optimizer.print_results()
ans = optimizer.results['x']

print(f"Peak: {peak / 10**6:.2f} MB")