from optiplex import Plex
import numpy as np
from sub_problem import make_sub_problem

N = 2
mission_ranges = np.linspace(1000, 3000, N)

subP_functions = []
for i in range(N):
    subPfunc = make_sub_problem(mission_ranges[i])
    subP_functions.append(subPfunc)


opt = Plex(blocks=subP_functions,
           constraint=constraint,
           x_init=v_init)

opt.solve(max_iter=100, 
          rho=1.2,
          tol=1e-7,
          ctol=1e-4)

