import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from subproblem_functions import make_subproblem
from optiplex import PlexC, Plex2, combo






num = 2 # number of operating conditions
sampler = LatinHypercube(d=2, seed=42)
samples = scale(sampler.random(num), l_bounds=[0.4, 180], u_bounds=[0.6, 220])
print(samples)

# generate subproblem functions
subPfuns = []
for i, (rho_atm, v_inf) in enumerate(samples): 
    subPfuns.append(make_subproblem(i, rho_atm, v_inf, num))


ns = 33 # number of spanwise panels
# alphas_init = np.zeros(num) # initial trim angles
# twists_init = np.ones((num, ns)) * np.deg2rad(5) # initial twist distributions
# thicknesses_init = np.ones((num, ns - 1)) * 0.01 # initial thickness distributions
# x_init = [alphas_init, twists_init, thicknesses_init]

alpha_i_init = np.array([0])
twist_i_init = np.ones(ns) * np.deg2rad(5)
thickness_i_init = np.ones(ns - 1) * 0.002

x_i_init = np.concatenate([alpha_i_init, twist_i_init, thickness_i_init])

x_init = [x_i_init for _ in range(num)]




def con(v_init):
    alphas = []
    twists = []
    thicknesses = []
    for i in range(num):
        x_init_i = v_init[i]
        alphas.append(x_init_i[0])
        twists.append(x_init_i[1:1 + ns])
        thicknesses.append(x_init_i[1 + ns:])

    twist_constraint = combo(twists) # modified combo to remove one pair
    thickness_constraint = combo(thicknesses) # modified combo to remove one pair

    return jnp.concatenate((twist_constraint, thickness_constraint))



opt = PlexC(subproblems=subPfuns,
            x_init=x_init,
            con=con,
            mu=10,
            max_mu=1e6,
            rho=1.5,
            tau=0.5,
            tol=1e-3, # outer loop feasibility
            eps=1e-3, # inner loop convergence
            )

opt.solve(max_outer_iter=2, 
          max_inner_iter=10,
          )