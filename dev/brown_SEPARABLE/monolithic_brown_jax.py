import numpy as np
import modopt as mo
import jax.numpy as jnp
import tracemalloc

# https://benchmarkfcns.info/doc/brownfcn.html

n = 10000 # dimension

tracemalloc.start()

# def jax_obj(v):
#     obj = 0
#     for i in range(n-1):
#         term1 = (v[i]**2)**(v[i+1]**2 + 1)
#         term2 = (v[i+1]**2)**(v[i]**2 + 1)
#         obj += term1 + term2
#     return obj

jax_obj = lambda v: jnp.sum((v[:-1]**2)**(v[1:]**2 + 1) + (v[1:]**2)**(v[:-1]**2 + 1))

# The function can be defined on any input domain but it is usually evaluated on xi∈[−1,4] for i=1,…,n.
x0 = np.ones((n,))
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 5000, 'ftol': 1e-8}, turn_off_outputs=True)

optimizer.solve()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

optimizer.print_results()
ans = optimizer.results['x']
# print(ans)

print(f"Peak: {peak / 10**6:.2f} MB")