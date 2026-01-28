import numpy as np
import modopt as mo
import jax.numpy as jnp
import tracemalloc

n = 100 # dimension

# needs a scaler at very large dimensions because the objective blows up
# but this necessitates a tighter convergence tolerance

# maybe I should modify the function to not blow up so much...

tracemalloc.start()

scaler = 1e-5
jax_obj = lambda v: scaler * (jnp.sum(v**2) + (0.5 * jnp.sum(jnp.arange(1, n + 1) * v))**2 + (0.5 * jnp.sum(jnp.arange(1, n + 1) * v))**4)

# The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d.
x0 = np.ones((n,)) * 0.1
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 5000, 'ftol': 1e-10}, turn_off_outputs=True)

optimizer.solve()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

optimizer.print_results()
ans = optimizer.results['x']
# print(ans)

print(f"Peak: {peak / 10**6:.2f} MB")