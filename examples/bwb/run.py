from optiplex import Plex
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import cstate
from subproblem_1 import subproblem_1
from subproblem_2 import subproblem_2
from model_1 import sim, design_variables, al_var_dict

# set the initial DV values from the SP1 sim
pitch = sim[design_variables['pitch'].variable]
half_ttop = sim[design_variables['half_ttop'].variable]
half_tweb = sim[design_variables['half_tweb'].variable]
wing_twist_coefficients = sim[design_variables['wing_twist_coefficients'].variable]
center_wing_half_span = sim[design_variables['center_wing_half_span'].variable]
transition_half_span = sim[design_variables['transition_half_span'].variable]
wing_half_span = sim[design_variables['wing_half_span'].variable]
center_wing_chord_stretch_coefficients = sim[design_variables['center_wing_chord_stretch_coefficients'].variable]
wing_root_chord = sim[design_variables['wing_root_chord'].variable]
wing_tip_chord = sim[design_variables['wing_tip_chord'].variable]
wing_sweep = sim[design_variables['wing_sweep'].variable]
transition_sweep = sim[design_variables['transition_sweep'].variable]
oversized_payload_translation_x = sim[design_variables['oversized_payload_translation_x'].variable]
oversized_payload_translation_z = sim[design_variables['oversized_payload_translation_z'].variable]
oversized_payload_rotation = sim[design_variables['oversized_payload_rotation'].variable]
cruise_trim_elevator_deflection = sim[design_variables['cruise_trim_elevator_deflection'].variable]
slack = sim[al_var_dict['slack']]

x_init = [pitch, half_ttop, 
          half_tweb, 
          wing_twist_coefficients, 
          center_wing_half_span, 
          transition_half_span, 
          wing_half_span, 
          center_wing_chord_stretch_coefficients, 
          wing_root_chord, 
          wing_tip_chord, 
          wing_sweep, 
          transition_sweep, 
          oversized_payload_translation_x, 
          oversized_payload_translation_z, 
          oversized_payload_rotation, 
          cruise_trim_elevator_deflection,
          slack]


# the global_con function returns the latest constraint values from the most recent subproblem
def global_con(x):
    print('global constraints: ', cstate.cstate)
    return cstate.cstate


opt = Plex(subproblems=[subproblem_1, subproblem_2],
           x_init=x_init,
           con=global_con,
           )

opt.solve(max_outer_iter=3,
          max_inner_iter=2,
          ATOL_out=1e-3, 
          RTOL_out=1e-3,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-3,
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