from optiplex import Plex, H5Plex
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import cstate
from subproblem_1 import subproblem_1
from subproblem_2 import subproblem_2

# load x_init from the pickle file
with open('examples/bwb/x_init.pkl', "rb") as f:
    x_init = pickle.load(f)
    x_init = x_init[:-1] # remove the slack and geonic for now

# # set the initial DV values from the SP1 sim
# pitch = sim[design_variables['pitch'].variable]
# half_ttop = sim[design_variables['half_ttop'].variable]
# half_tweb = sim[design_variables['half_tweb'].variable]
# wing_twist_coefficients = sim[design_variables['wing_twist_coefficients'].variable]
# center_wing_half_span = sim[design_variables['center_wing_half_span'].variable]
# transition_half_span = sim[design_variables['transition_half_span'].variable]
# wing_half_span = sim[design_variables['wing_half_span'].variable]
# center_wing_chord_stretch_coefficients = sim[design_variables['center_wing_chord_stretch_coefficients'].variable]
# wing_root_chord = sim[design_variables['wing_root_chord'].variable]
# wing_tip_chord = sim[design_variables['wing_tip_chord'].variable]
# wing_sweep = sim[design_variables['wing_sweep'].variable]
# transition_sweep = sim[design_variables['transition_sweep'].variable]
# oversized_payload_translation_x = sim[design_variables['oversized_payload_translation_x'].variable]
# oversized_payload_translation_z = sim[design_variables['oversized_payload_translation_z'].variable]
# oversized_payload_rotation = sim[design_variables['oversized_payload_rotation'].variable]
# cruise_trim_elevator_deflection = sim[design_variables['cruise_trim_elevator_deflection'].variable]

# x_init = [pitch, 
#           half_ttop, 
#           half_tweb, 
#           wing_twist_coefficients, 
#           center_wing_half_span, 
#           transition_half_span, 
#           wing_half_span, 
#           center_wing_chord_stretch_coefficients, 
#           wing_root_chord, 
#           wing_tip_chord, 
#           wing_sweep, 
#           transition_sweep, 
#           oversized_payload_translation_x, 
#           oversized_payload_translation_z, 
#           oversized_payload_rotation, 
#           cruise_trim_elevator_deflection,]

# # save x_init for later use
# with open('examples/bwb/x_init.pkl', "wb") as f:
#     pickle.dump(x_init, f)
# exit()


# the solution from the monolithic problem
solution = np.array([5.000000000000000000e+00, 3.171801070007730927e-01, 4.258407738921992092e-01, 2.148450654742703136e+00, 5.298824025117689906e+00, 5.009175638636133243e+00, 3.578328778488450101e+00, 2.304196056839215245e+00, 1.317684182368547008e+00,
                      5.678887534024594785e-01, 1.349987368368475438e-01, 3.739531082775680026e-01, 5.124605762018374921e-01, 6.018793069388612693e-01, 1.636566884380767695e+00, 1.159719490516270657e+00, 8.966148236070672350e-01, 9.236627815410372033e-01,
                        3.580846560119934630e-01, 2.268567878619339728e-01, 1.000000000000000056e-01, -1.997815673298932504e+00, -1.522672421686179334e+00, 3.000000000000000000e+00, 3.000000000000000000e+00, 1.810991781698371383e+01, 7.039180575579780941e-01,
                          -2.899443452376383235e+00, -2.352346192065830621e+00, -2.664635319363378052e+00, 5.084006503034280477e+00, 1.630155573193770913e+00, 4.002663340699021965e+00, 2.904719345075986059e+00, 1.000000000000000000e+01, 0.000000000000000000e+00,
                            6.404653491181701619e-15, 5.139628064721836198e-01])

# the global_con function returns the latest constraint values from the most recent subproblem
def global_con(x):
    print('global constraints: ', cstate.cstate)
    return cstate.cstate


# opt = Plex(subproblems=[subproblem_1, subproblem_2],
#            x_init=x_init,
#            con=global_con,
#            )

opt = H5Plex(subproblems=[subproblem_1, subproblem_2],
             x_init=x_init,
             con=global_con,
             path="examples/bwb/checkpoint.h5",
             solution=solution,
             )

opt.solve(max_outer_iter=300,
          max_inner_iter=2,
          ATOL_out=1e-4, 
          RTOL_out=1e-4,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-3,
          rho=1.2,
          mu=10.0,
          )

solution = opt.x

# save opt.history to a pickle file
with open('examples/bwb/history.pkl', "wb") as f:
    pickle.dump(opt.history, f)

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