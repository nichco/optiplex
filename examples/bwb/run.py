from optiplex import H5Plex
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import cstate
from subproblem_1 import subproblem_1
from subproblem_2 import subproblem_2

# # load x_init from the pickle file
# with open('examples/bwb/x_init.pkl', "rb") as f:
#     x_init = pickle.load(f)
#     x_init = x_init[:-1] # remove the slack and geonic for now


# with h5py.File('examples/bwb/checkpoint.h5', 'r') as f:
#     latest_iteration = sorted((int(key) for key in f.keys()))[-1]
#     x = f[str(latest_iteration)]['x'][()]

# x_init = [x[0:1],
#           x[1:11],
#           x[11:21],
#           x[21:23],
#           x[23:24],
#           x[24:25],
#           x[25:26],
#           x[26:30],
#           x[30:31],
#           x[31:32],
#           x[32:33],
#           x[33:34],
#           x[34:35],
#           x[35:36],
#           x[36:37],
#           x[37:38]]

# print(x_init)
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



# ###############################################################################################################################
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
#           cruise_trim_elevator_deflection] # try this to speed things up??
# ###############################################################################################################################

###############################################################################################################################
x_init = [np.array([5.0]),
          half_ttop + 0.001,
          half_tweb + 0.001,
          np.array([-2e+00, -2e+00]),
          np.array([3]),
          np.array([3]),
          np.array([1.5e+01]),
          np.array([-5.5e-01, -4e+00, -3e+00, -3.5e+00]),
          np.array([5]),
          np.array([1.5]),
          np.array([40]),
          np.array([30]),
          np.array([9]), # WAS 10 for checkpoint 7
          np.array([0]),
          np.array([0]),
          np.array([5])] # try this to speed things up??
###############################################################################################################################



# the global_con function returns the latest constraint values from the most recent subproblem
def global_con(x):
    print('global constraints: ', cstate.cstate)
    return cstate.cstate


opt = H5Plex(subproblems=[subproblem_1, subproblem_2],
             x_init=x_init,
             con=global_con,
             path="examples/bwb/checkpoint7.h5",
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

# checkpoint 6
# opt.solve(max_outer_iter=300,
#           max_inner_iter=2,
#           ATOL_out=1e-5, 
#           RTOL_out=1e-5,
#           ATOL_in=1e-2, 
#           RTOL_in=1e-2,
#           ATOL_feas=1e-4,
#           rho=1.2,
#           mu=100.0,
#           )

# checkpoint 7
opt.solve(max_outer_iter=300,
          max_inner_iter=2,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-4,
          rho=1.2,
          mu=30.0,
          )

solution = opt.x

# save opt.history to a pickle file
with open('examples/bwb/history.pkl', "wb") as f:
    pickle.dump(opt.history, f)