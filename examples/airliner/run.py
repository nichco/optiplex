from optiplex import Plex2
import numpy as np
from sub_p_funs import make_sub_problem
from typing import List
from optiplex import combo
import jax.numpy as jnp
import tracemalloc

tracemalloc.start()

n_eta = 50
n_theta = 50

N = 2
rvals = np.linspace(1e6, 6e6, N) # mission range values

# generate subproblem functions
subPfuns, data = [], [] # data contains optimization times
for i, r in enumerate(rvals): subPfuns.append(make_sub_problem(i, r, N, data))

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
x0 = np.concatenate((np.linspace(0.6, 0.5, n_eta),
                     np.full((n_theta), np.deg2rad(6)),
                     np.array([16750.0]), 
                     np.array([22]), 
                     np.array([80]),
                     np.array([2000.0]))
                     )
v_init = [x0] * N



# opt = Plex(subproblems=subPfuns,
#            x_init=v_init,
#            con=constraint,
#            )

# opt.solve(max_outer_iter=1,#300,
#           max_inner_iter=1,#3,
#           ATOL_out=1e-4, 
#           RTOL_out=1e-4,
#           ATOL_in=1e-2, 
#           RTOL_in=1e-2,
#           ATOL_feas=1e-4,
#           rho=1.2,
#           mu=1.0
#           )

opt = Plex2(subproblems=subPfuns,
            x_init=v_init,
            con=constraint,
            mu=1,
            max_mu=1e6,
            rho=1.5,
            tau=0.5,
            tol=0.5e-3, # outer loop feasibility
            eps=1e-2, # initial inner loop convergence
            eta=1e-4, # final inner loop convergence
            )

opt.solve(max_outer_iter=100, 
          max_inner_iter=5,
          )


current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory usage: {peak / 10**6} MB")
tracemalloc.stop()

AR_vals, S_vals = [], []
for i in range(N):
    xi = opt.x[i]
    eta_i = xi[:n_eta]
    theta_i = xi[n_eta:-4]
    tf_i = xi[-4]
    AR_i = xi[-3]
    S_i = xi[-2]
    fuel_i = xi[-1]

    AR_vals.append(np.atleast_1d(AR_i))
    S_vals.append(np.atleast_1d(S_i))

print('AR values: ', np.concatenate(AR_vals))
print('S values: ', np.concatenate(S_vals))

print('Total Time (s): ', opt.tf)
print('Optimization time (s): ', data[-1])





solution = np.load('examples/airliner/airliner_solution_N2.npz')['x']






# # old ipopt data
# # num_subPs = np.array([2, 4, 6, 8, 10])
# # time_data = np.array([298.4, 534.8, 937.5, 1551.2, 2227.4])




# # new slsqp data
# num_subPs = np.array([2, 4, 6, 8, 10])
# time_data = np.array([99.3, 145.9, 194.8, 284.0, 343.8])


# memory = np.array([100.3, 129.6, 173.5, 204.2, 235.0])