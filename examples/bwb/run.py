from optiplex import Plex
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from subproblem_1 import subproblem_1
from subproblem_2 import subproblem_2
from global_con import global_con

# initial DV values
pitch = 
half_ttop = 
half_tweb = 
center_wing_twist_dv = 
wing_twist_coefficients = 
wing_twist_coefficients = 
center_wing_half_span = 
transition_half_span = 
wing_half_span = 
center_wing_chord_stretch_coefficients = 
wing_root_chord = 
wing_tip_chord = 
wing_sweep = 

x_init = np.concatenate()




opt = Plex(subproblems=[subproblem_1, subproblem_2],
           x_init=x_init,
           con=global_con,
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=10,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-5,
          rho=1.2,
          mu=10.0,
          )

solution = opt.x



# monolithic_solution = np.load('examples/bwb/solution.npz')
# x_star = np.concatenate([monolithic_solution['twist'], monolithic_solution['thickness']])

# history_vecs = [np.concatenate(h[:2]) for h in opt.history]
# error = [np.linalg.norm((x - x_star) / x_star) for x in history_vecs]

# plt.semilogy(opt.x_time, error)
# plt.xlabel('Time (s)')
# plt.ylabel('Relative error')
# plt.show()

# save error history and mu history and x_time
# np.savez('examples/bwb/history.npz', error=error, mu_history=opt.mu_history, x_time=opt.x_time)