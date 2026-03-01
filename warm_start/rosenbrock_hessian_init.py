import numpy as np
import modopt as mo
import jax.numpy as jnp
import jax

n = 4 # dimension


jax_obj = lambda v: jnp.sum(100 * (v[1:] - v[:-1]**2)**2 + (1 - v[:-1])**2)

# compute hessian with jax
hessian = jax.hessian(jax_obj)(jnp.zeros((n,)))
print("Hessian at zero:", hessian)

# guess = np.array([-1.2, 1] * (n // 2))
# x0 = guess
x0 = np.zeros((n,))
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 7000, 'ftol': 1e-7}, turn_off_outputs=True)
# optimizer = mo.PySLSQP(jaxprob, solver_options={'maxiter': 7000, 'ftol': 1e-7}, turn_off_outputs=True)
# optimizer = mo.InteriorPoint(jaxprob, recording=False, turn_off_outputs=True, maxiter=6000, opt_tol=1e-7, feas_tol=1e-7)
# optimizer = mo.OpenSQP(jaxprob, maxiter=6000, opt_tol=1e-7, turn_off_outputs=True)

optimizer.solve()


optimizer.print_results()
# ans = optimizer.results['x']