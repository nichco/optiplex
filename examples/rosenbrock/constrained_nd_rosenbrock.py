import numpy as np
import modopt as mo
import jax.numpy as jnp
import tracemalloc

# https://www.sfu.ca/~ssurjano/rosen.html

n = 100 # dimension

tracemalloc.start()

jax_obj = lambda v: jnp.sum(100 * (v[1:] - v[:-1]**2)**2 + (1 - v[:-1])**2)

jax_con = lambda v: jnp.sum(v**2).flatten()

x0 = np.zeros((n,))
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, jax_con=jax_con, xl=-np.inf, xu=np.inf, cl=-np.inf, cu=1)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 7000, 'ftol': 1e-7}, turn_off_outputs=True)
# optimizer = mo.InteriorPoint(jaxprob, recording=False, turn_off_outputs=True, maxiter=6000, opt_tol=1e-7, feas_tol=1e-7)

optimizer.solve()

current, peak = tracemalloc.get_traced_memory()

tracemalloc.stop()

optimizer.print_results()
ans = optimizer.results['x']
print('solution: ', ans)
print(f"Radius: {np.linalg.norm(ans):.4f}")

print(f"Peak: {peak / 10**6:.2f} MB")