from optiplex import H5Plex
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import cstate
from subproblem_1 import subproblem_1
from subproblem_2 import subproblem_2
import h5py

# # load x_init from the pickle file
# with open('examples/bwb/x_init.pkl', "rb") as f:
#     x_init = pickle.load(f)
#     x_init = x_init[:-1] # remove the slack and geonic for now


with h5py.File('examples/bwb/checkpoint.h5', 'r') as f:
    latest_iteration = sorted((int(key) for key in f.keys()))[-1]
    x = f[str(latest_iteration)]['x'][()]

x_init = [x[0:1],
          x[1:11],
          x[11:21],
          x[21:23],
          x[23:24],
          x[24:25],
          x[25:26],
          x[26:30],
          x[30:31],
          x[31:32],
          x[32:33],
          x[33:34],
          x[34:35],
          x[35:36],
          x[36:37],
          x[37:38]]

# print(x_init)
# exit()



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
pitch = np.array([5.0])
half_ttop = 1e-2 * np.array([3.471430464523918569e-01, 4.710321324047168634e-01, 2.346720892339240194e+00, 5.141665869436192970e+00, 4.671710754773619634e+00, 3.308906469280518792e+00, 2.260935542357682682e+00, 1.267027675320232083e+00, 5.310302445802553839e-01, 1.205186623337621693e-01])
half_tweb = 1e-2 * np.array([3.929994704531686867e-01, 5.541758863744186137e-01, 6.039841197772257697e-01, 1.381458071395717635e+00, 1.123180483752061587e+00, 8.978255449258085719e-01, 7.057174470185272330e-01, 3.472195393658423224e-01, 1.716305123422086076e-01, 1.000000000000000056e-01,])
wing_twist_coefficients = np.array([-1.864996758674682065e+00, -1.887563731814386303e+00])
center_wing_half_span = np.array([3.000000000000000444e+00])
transition_half_span = np.array([3.000000000000000444e+00])
wing_half_span = np.array([1.771681807724542068e+01])
center_wing_chord_stretch_coefficients = np.array([-5.574281883182186093e-01, -4.053422828244272580e+00, -3.341042257413223471e+00, -3.548134619452329108e+00])
wing_root_chord = np.array([5.114889837458465927e+00])
wing_tip_chord = np.array([1.694647233448457246e+00])
wing_sweep = np.array([3.966501558870209276e+01])
transition_sweep = np.array([3.143469885871966518e+01])
oversized_payload_translation_x = np.array([9.999999999999998224e+00])
oversized_payload_translation_z = np.array([0.000000000000000000e+00])
oversized_payload_rotation = np.array([-3.514884289414112108e-14])
cruise_trim_elevator_deflection = np.array([5.084434597044037440])

solution = np.concatenate([pitch,
                           half_ttop,
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
                           cruise_trim_elevator_deflection])

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

# original running version:
# opt.solve(max_outer_iter=300,
#           max_inner_iter=2,
#           ATOL_out=1e-4, 
#           RTOL_out=1e-4,
#           ATOL_in=1e-2, 
#           RTOL_in=1e-2,
#           ATOL_feas=1e-3,
#           rho=1.2,
#           mu=50.0,
#           )

# new version:
# opt.y = np.ones(4) # i might not need this anymore, now that i fixed the csdl error.

# opt.solve(max_outer_iter=300,
#           max_inner_iter=2,
#           ATOL_out=1e-4, 
#           RTOL_out=1e-4,
#           ATOL_in=1e-2, 
#           RTOL_in=1e-2,
#           ATOL_feas=1e-3,
#           rho=1.2,
#           mu=100.0,
#           )
opt.solve(max_outer_iter=300,
          max_inner_iter=2,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-3,
          rho=1.2,
          mu=60.0,
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