import numpy as np
import modopt as mo
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

n = 1000


jax_obj = lambda v: jnp.sum(100 * (v[1:] - v[:-1]**2)**2 + (1 - v[:-1])**2)


x0 = np.zeros((n,))

# compute hessian with jax
H = jax.hessian(jax_obj)(x0)

# eig clip
e, v = np.linalg.eig(H)
eps = 1e-4
e = np.clip(e, a_min=eps, a_max=None)
H = v.dot(np.diag(e)).dot(v.T)
H = (H + H.T) / 2 # force symmetry


x0 = np.zeros((n,))
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

# optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 7000, 'ftol': 1e-7}, turn_off_outputs=True)
# optimizer = mo.InteriorPoint(jaxprob, recording=False, turn_off_outputs=True, maxiter=6000, opt_tol=1e-7, feas_tol=1e-7)

# optimizer = mo.OpenSQP(jaxprob, maxiter=6000, opt_tol=1e-7, turn_off_outputs=True)
optimizer = mo.OpenSQP(jaxprob, maxiter=6000, opt_tol=1e-7, turn_off_outputs=True, bfgs_init_lag_hess=H)

optimizer.solve()

optimizer.print_results()


# nothing (n=100)
# nfev = ngev = 778

# eig_clip (n=100)
# nfev = ngev = 490

# eig flip (n=100)
# nfev = ngev = 490




# nothing (n=300)
# nfev = ngev = 2302

# eig clip (n=300)
# nfev = ngev = 1395