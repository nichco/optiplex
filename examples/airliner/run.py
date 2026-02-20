from optiplex import Plex
import numpy as np
from sub_p_funs import make_sub_problem
from typing import List
from optiplex import combo
import jax.numpy as jnp
from meta_data import Params

N = 2
rvals = np.linspace(1e6, 6e6, N) # mission range values

subPfuns = []
for i, r in enumerate(rvals): subPfuns.append(make_sub_problem(i, r, N))


# pair-wise differences formulation
def constraint(x_init: List[np.ndarray]) -> jnp.ndarray:
    
    global_vars = x_init[:2*N]
    AR_list = global_vars[:N]
    S_list = global_vars[N:]

    AR_constraint = combo(AR_list)
    S_constraint = combo(S_list)
    return jnp.concatenate((AR_constraint, S_constraint))



# v_init = [AR_1, ..., AR_N, S_1, ..., S_N, d_1, ..., d_N]

v_init = [np.array([22])] * N + [np.array([75])] * N
for r in rvals:
    d_i = np.concatenate((Params[r].eta, Params[r].theta, Params[r].tf, Params[r].fuel))
    v_init.append(d_i)


opt = Plex(subproblems=subPfuns,
           x_init=v_init,
           con=constraint,
           )

opt.solve(max_outer_iter=2,
          max_inner_iter=1,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-3,
          rho=1.05,
          mu=0.1
          )

# print('Solution: ', opt.x)
print('Time (s): ', opt.time)