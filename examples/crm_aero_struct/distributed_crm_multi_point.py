import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from subproblem_functions import make_subproblem
from optiplex import PlexC, Plex2






num = 15 # number of operating conditions
sampler = LatinHypercube(d=2, seed=42)
samples = scale(sampler.random(num), l_bounds=[0.4, 180], u_bounds=[0.6, 220])
print(samples)

# generate subproblem functions
subPfuns = []
for i, (rho_atm, v_inf) in enumerate(samples): 
    subPfuns.append(make_subproblem(i, rho_atm, v_inf, num))








opt = PlexC(subproblems=subPfuns,
            x_init=x_init,
            con=con,
            mu=10,
            max_mu=1e6,
            rho=1.5,
            tau=0.5,
            tol=1e-3, # outer loop feasibility
            eps=1e-5, # inner loop convergence
            )

opt.solve(max_outer_iter=100, 
          max_inner_iter=100,
          )