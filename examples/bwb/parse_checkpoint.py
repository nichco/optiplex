import numpy as np
import h5py
import matplotlib.pyplot as plt


path = 'examples/bwb/checkpoint_copy_6.h5'

with h5py.File(path, 'r') as f:
	iteration_ids = sorted((int(key) for key in f.keys()))

	x_history = [f[str(iteration_id)]['x'][()] for iteration_id in iteration_ids]
	error_history = [f[str(iteration_id)]['error'][()] for iteration_id in iteration_ids]

# x_star = np.array([5.000000000000000000e+00, 3.471430464523918569e-01, 4.710321324047168634e-01, 2.346720892339240194e+00, 5.141665869436192970e+00, 4.671710754773619634e+00, 3.308906469280518792e+00, 
# 					 2.260935542357682682e+00, 1.267027675320232083e+00, 5.310302445802553839e-01, 1.205186623337621693e-01, 3.929994704531686867e-01, 5.541758863744186137e-01, 6.039841197772257697e-01, 
# 					 1.381458071395717635e+00, 1.123180483752061587e+00, 8.978255449258085719e-01, 7.057174470185272330e-01, 3.472195393658423224e-01, 1.716305123422086076e-01, 1.000000000000000056e-01,
# 					 -1.864996758674682065e+00, -1.887563731814386303e+00, 3.000000000000000444e+00, 3.000000000000000444e+00, 1.771681807724542068e+01, -5.574281883182186093e-01, -4.053422828244272580e+00,
# 					 -3.341042257413223471e+00, -3.548134619452329108e+00, 5.114889837458465927e+00, 1.694647233448457246e+00, 3.966501558870209276e+00, 3.143469885871966518e+00, 9.999999999999998224e+00,
# 					 0.000000000000000000e+00, -3.514884289414112108e-14, 5.084434597044037440e-01])

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

x_star = np.concatenate([pitch,
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



denominator = np.where(np.abs(x_star) > 1e-12, x_star, 1.0) # prevent divide-by-zero errors
error = [np.linalg.norm((x_history[i] - x_star) / denominator) for i in range(len(x_history))]

# print(error)

# increase numpy printing line width so that the full x_star vector is printed on one line
np.set_printoptions(linewidth=400, precision=4, suppress=True)

print(x_star)
print(x_history[-1])
# print(x_history[-1] - x_history[-2])


# fig, ax1 = plt.subplots(figsize=(4, 3))
plt.figure(figsize=(4, 3))

plt.plot(error, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Error')

plt.savefig('bwb_convergence.png', bbox_inches='tight', transparent=True, dpi=600)
plt.show()