import numpy as np
from optiplex import PlexC, Plex2, combo
from scipy.stats.qmc import LatinHypercube, scale
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

# sample the space of uncertain parameters using Latin hypercube sampling
# uncertain parameters: g, mu_cart, mu_pole
g_u = 9.81 + 0.1
g_l = 9.81 - 0.1
mu_cart_u = 0.03 + 0.01
mu_cart_l = 0.03 - 0.01
mu_pole_u = 0.03 + 0.01
mu_pole_l = 0.03 - 0.01
N = 5
sampler = LatinHypercube(d=3, seed=0)
samples = scale(sampler.random(N), 
                l_bounds=[g_l, mu_cart_l, mu_pole_l], 
                u_bounds=[g_u, mu_cart_u, mu_pole_u])
# samples = np.array([[9.81, 0.03, 0.03], [9.81, 0.03, 0.03]])
print(samples)




subPfuns, opt_time = [], []
for i in range(N):
    subPfuns.append(make_subproblem(i, N, opt_time, samples))







def constraint(v):

    return


opt = Plex2(subproblems=subPfuns,
           con=constraint,
           tol=1e-5, # feasibility tolerance
           rho=1.1, # penalty increase factor
           mu=1.0, # initial penalty parameter
           max_mu=1e6, # maximum penalty parameter
           x_init=v_init,
           eps=1e-2, # initial inner loop tolerance
           eta=1e-5, # final inner loop tolerance
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=10,
          )