from optiplex import Plex
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from subproblem_1 import subproblem_1
from subproblem_2 import subproblem_2
from global_con import global_con

from model_1 import sim, design_variables

# initial DV values
pitch = sim[design_variables['pitch'].variable.value]
half_ttop = sim[design_variables['half_ttop'].variable.value]
half_tweb = sim[design_variables['half_tweb'].variable.value]
wing_twist_coefficients = sim[design_variables['wing_twist_coefficients'].variable.value]
center_wing_half_span = sim[design_variables['center_wing_half_span'].variable.value]
transition_half_span = sim[design_variables['transition_half_span'].variable.value]
wing_half_span = sim[design_variables['wing_half_span'].variable.value]
center_wing_chord_stretch_coefficients = sim[design_variables['center_wing_chord_stretch_coefficients'].variable.value]
wing_root_chord = sim[design_variables['wing_root_chord'].variable.value]
wing_tip_chord = sim[design_variables['wing_tip_chord'].variable.value]
wing_sweep = sim[design_variables['wing_sweep'].variable.value]
transition_sweep = sim[design_variables['transition_sweep'].variable.value]
oversized_payload_translation_x = sim[design_variables['oversized_payload_translation_x'].variable.value]
oversized_payload_translation_z = sim[design_variables['oversized_payload_translation_z'].variable.value]
oversized_payload_rotation = sim[design_variables['oversized_payload_rotation'].variable.value]
cruise_trim_elevator_deflection = sim[design_variables['cruise_trim_elevator_deflection'].variable.value]

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
          cruise_trim_elevator_deflection]




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