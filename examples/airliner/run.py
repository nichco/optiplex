from optiplex import Plex
import numpy as np
from sub_p_funs import make_sub_problem
from typing import List
from optiplex import combo
import jax.numpy as jnp
from meta_data import Params

N = 2
rvals = np.linspace(1e6, 6e6, N) # mission range values

# generate subproblem functions
subPfuns = []
for i, r in enumerate(rvals): subPfuns.append(make_sub_problem(i, r, N))

# pair-wise differences formulation
def constraint(x_init: List[np.ndarray]) -> jnp.ndarray:
    global_vars = x_init[:2*N]
    AR_list = global_vars[:N]
    S_list = global_vars[N:]

    return jnp.concatenate((combo(AR_list), combo(S_list)))

# initial guesses for all variables
# v_init = [AR_1, ..., AR_N, S_1, ..., S_N, d_1, ..., d_N]
v_init = [np.array([22])] * N + [np.array([75])] * N
for r in rvals:
    v_init.append(np.concatenate((Params[r].eta, Params[r].theta, Params[r].tf, Params[r].fuel)))


opt = Plex(subproblems=subPfuns,
           x_init=v_init,
           con=constraint,
           )

opt.solve(max_outer_iter=10,
          max_inner_iter=1,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-3,
          rho=1.1,
          mu=1.0
          )

# print('Solution: ', opt.x)
print('Time (s): ', opt.time)