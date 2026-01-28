import numpy as np
import modopt as mo
import jax.numpy as jnp
import tracemalloc

n = 100 # dimension

tracemalloc.start()

jax_obj = lambda v: jnp.sum(v**2) + (0.5 * jnp.sum(jnp.arange(1, v.size + 1) * v))**2 + (0.5 * jnp.sum(jnp.arange(1, v.size + 1) * v))**4


x0 = np.ones((n,))
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 5000, 'ftol': 1e-7}, turn_off_outputs=True)

optimizer.solve()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

optimizer.print_results()
ans = optimizer.results['x']

print(f"Peak: {peak / 10**6:.2f} MB")