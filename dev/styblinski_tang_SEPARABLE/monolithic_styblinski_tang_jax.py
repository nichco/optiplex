import numpy as np
import modopt as mo
import jax.numpy as jnp
import tracemalloc

# https://www.sfu.ca/~ssurjano/stybtang.html

n = 10000 # dimension
# n=10000, t=56.8

tracemalloc.start()

# requires n to be in the thousands before it starts to slow down...
# The function is usually evaluated on the hypercube xi ∈ [-5, 5], for all i = 1, …, d.
jax_obj = lambda v: 0.5 * jnp.sum(v**4 - 16 * v**2 + 5 * v)
    
x0 = np.ones((n,)) * -1
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 5000, 'ftol': 1e-7}, turn_off_outputs=True)

optimizer.solve()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

optimizer.print_results()
ans = optimizer.results['x']
# print(ans)

print(f"Peak: {peak / 10**6:.2f} MB")