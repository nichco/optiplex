from optiplex import Plex
import numpy as np
from sub_p_funs import make_sub_problem
from typing import List
from optiplex import combo
import jax.numpy as jnp

nu = 300

N = 2
rvals = np.linspace(1e6, 6e6, N) # mission range values

# generate subproblem functions
subPfuns = []
for i, r in enumerate(rvals): subPfuns.append(make_sub_problem(i, r, N))

# pair-wise differences formulation
def constraint(x_init: List[np.ndarray]) -> jnp.ndarray:

    AR_list, S_list = [], []
    for j in range(N):
        AR_j = x_init[j][-3]
        S_j = x_init[j][-2]

        AR_list.append(AR_j)
        S_list.append(S_j)

    AR_constraint, S_constraint = combo(AR_list), combo(S_list)

    return jnp.concatenate((AR_constraint, S_constraint)) / 1e2

# initial guesses for all variables
# order of vars: eta_i, theta_i, tf_i, AR_i, S_i, fuel_i
x0 = np.concatenate((np.linspace(0.6, 0.5, nu),
                     np.full((nu), np.deg2rad(6)),
                     np.array([16750.0]), 
                     np.array([22]), 
                     np.array([80]),
                     np.array([2000.0]))
                     )
v_init = [x0] * N



opt = Plex(subproblems=subPfuns,
           x_init=v_init,
           con=constraint,
           )

opt.solve(max_outer_iter=3,
          max_inner_iter=1,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-3,
          rho=1.05,
          mu=0.5
          )

# print('Solution: ', opt.x)
print('Time (s): ', opt.time)