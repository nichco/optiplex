import numpy as np

n_cruise_trim = 1
n_CM_cg_cruise_nominal = 1
n_static_margin = 1
n_beam_max_stress_S1_SF = 1
n_oversized_payload_sdf_values = 26 # UPDATE IF THE NUMBER CHANGES!!!

nc = n_cruise_trim + n_CM_cg_cruise_nominal + n_static_margin + n_beam_max_stress_S1_SF + n_oversized_payload_sdf_values


# placeholder for global constraint vals
cstate = np.zeros(nc)