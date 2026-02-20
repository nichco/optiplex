from optiplex import Plex
import numpy as np
from sub_p_funs import make_sub_problem
from typing import List
from optiplex import combo
import jax.numpy as jnp

# N = 2
# mission_ranges = np.linspace(1.5e6, 3e6, N)

# subP_functions = []
# for i in range(N):
#     subPfunc = make_sub_problem(mission_ranges[i], i, N)
#     subP_functions.append(subPfunc)


# nu = 60
# eta0 = np.linspace(0.6, 0.2, nu)
# theta0 = np.linspace(np.deg2rad(3), np.deg2rad(1), nu)
# tf0 = np.array([7000.0])
# b0 = np.array([32.0])

# v_init = []
# # automate the construction of v_init for changing N
# for i in range(N): 
#     v_init.append(eta0)
#     v_init.append(theta0)
#     v_init.append(tf0)
#     v_init.append(b0)

N = 4
rvals = np.linspace(1e6, 6e6, N) # mission range values

subPfuns = []
for i, r in enumerate(rvals):
    subPfuns.append(make_sub_problem(r))




# pair-wise differences formulation
def constraint(x_init: List[np.ndarray]) -> jnp.ndarray:
    
    # l1 = x_init[0] # need to expand for changing N
    # l2 = x_init[1] 
    # mp1 = x_init[2]
    # mp2 = x_init[3]

    AR_list = [x_init[index] for ...]
    S_list = [x_init[index] for ...]

    AR_constraint = combo(AR_list)
    S_constraint = combo(S_list)
    return jnp.concatenate((AR_constraint, S_constraint))



opt = Plex(subproblems=subPfuns,
           x_init=v_init,
           con=constraint,
           )

opt.solve(max_outer_iter=10,
          max_inner_iter=2,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-3,
          rho=1.05
          )

# print('Solution: ', opt.x)
print('Time (s): ', opt.time)