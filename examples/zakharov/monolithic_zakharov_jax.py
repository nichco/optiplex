import numpy as np
import modopt as mo
import jax.numpy as jnp
import tracemalloc

# https://www.sfu.ca/~ssurjano/zakharov.html

n = 6000 # dimension

tracemalloc.start()

# # original Zakharov function (blows up at high dimensions)
# def jax_obj(v):
#     s = 0.5 * jnp.sum(jnp.arange(1, n + 1) * v)
#     return jnp.sum(v**2) + s**2 + s**4

# # modified Zakharov function (normalized by dimension)
# def jax_obj(v):
#     s = 0.5 * jnp.sum(jnp.arange(1, n + 1) * v) / n
#     return jnp.sum(v**2) + s**2 + s**4

# modified Zakharov function (normalized by dimension)
def jax_obj(v):
    s2 = 0.5 * jnp.sum(jnp.arange(1, n + 1) * v) / n
    s4 = 0.5 * jnp.sum(jnp.arange(1, n + 1) * v) / n**2
    return jnp.sum(v**2) + s2**2 + s4**4


# The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d.
x0 = np.ones((n,)) * 0.1
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-8}, turn_off_outputs=True)

optimizer.solve()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

optimizer.print_results()
ans = optimizer.results['x']
# print(ans)

print(f"Peak: {peak / 10**6:.2f} MB")