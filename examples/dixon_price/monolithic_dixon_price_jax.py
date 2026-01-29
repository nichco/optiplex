import numpy as np
import modopt as mo
import jax.numpy as jnp
import tracemalloc

# https://www.sfu.ca/~ssurjano/dixonpr.html

n = 1000 # dimension

tracemalloc.start()

jax_obj = lambda v: (v[0] - 1)**2 + jnp.sum(jnp.arange(2, n + 1) * (2 * v[1:]**2 - v[:-1])**2)

# The function is usually evaluated on the hypercube xi ∈ [-10, 10], for all i = 1, …, d.
x0 = np.ones((n,))
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 5000, 'ftol': 1e-7}, turn_off_outputs=True)
# optimizer = mo.InteriorPoint(jaxprob, recording=False, turn_off_outputs=True, maxiter=6000, opt_tol=1e-7, feas_tol=1e-7)

optimizer.solve()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

optimizer.print_results()
ans = optimizer.results['x']

print(f"Peak: {peak / 10**6:.2f} MB")