import numpy as np
import modopt as mo
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

n = 1000

jax_obj = lambda v: (v[0] - 1)**2 + jnp.sum(jnp.arange(2, n + 1) * (2 * v[1:]**2 - v[:-1])**2)

# The function is usually evaluated on the hypercube xi ∈ [-10, 10], for all i = 1, …, d.
x0 = np.ones((n,))


H = np.array(jax.hessian(jax_obj)(x0))

H = (H + H.T) / 2 # force symmetry

e, v = np.linalg.eigh(H)

eps = 1e-5
e = np.clip(e, a_min=eps, a_max=None) # clip eigenvalues to ensure positive definiteness

H = v @ np.diag(e) @ v.T # reconstruct hessian with the clipped eigenvalues

H = (H + H.T) / 2 # force symmetry


jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

# optimizer = mo.OpenSQP(jaxprob, maxiter=6000, opt_tol=1e-7, turn_off_outputs=True)
optimizer = mo.OpenSQP(jaxprob, maxiter=6000, opt_tol=1e-7, turn_off_outputs=True, bfgs_init_lag_hess=H)

optimizer.solve()

optimizer.print_results()


# with hessian init (n=1000)
# nfev = 286, ngev = 227

# without hessian init (n=1000)
# nfev = 11697, ngev = 6851