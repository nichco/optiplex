import numpy as np
import h5py
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=400, precision=4, suppress=True)

path = 'examples/bwb/checkpoint copy.h5'
# path = 'examples/bwb/checkpoint10 copy.h5'

with h5py.File(path, 'r') as f:
    iteration_ids = sorted((int(key) for key in f.keys()))

    x_history = [f[str(iteration_id)]['x'][()] for iteration_id in iteration_ids]
    # error_history = [f[str(iteration_id)]['error'][()] for iteration_id in iteration_ids]
    mu_history = [f[str(iteration_id)]['mu'][()] for iteration_id in iteration_ids]
    y_history = [f[str(iteration_id)]['y'][()] for iteration_id in iteration_ids]

print('last x: ', x_history[-1])

xa = x_history[-1]
pitch_a = xa[0]
half_ttop_a = xa[1:11]
half_tweb_a = xa[11:21]
wing_twist_coefficients_a = xa[21:23]
center_wing_half_span_a = xa[23:24]
transition_half_span_a = xa[24:25]
wing_half_span_a = xa[25:26]
center_wing_chord_stretch_coefficients_a = xa[26:30]
wing_root_chord_a = xa[30:31]
wing_tip_chord_a = xa[31:32]
wing_sweep_a = xa[32:33]
transition_sweep_a = xa[33:34]
oversized_payload_translation_x_a = xa[34:35]
oversized_payload_translation_z_a = xa[35:36]
oversized_payload_rotation_a = xa[36:37]
cruise_trim_elevator_deflection_a = xa[37:38]



# pitch = np.array([5.0])
# half_ttop = 1e-2 * np.array([3.471430464523918569e-01, 4.710321324047168634e-01, 2.346720892339240194e+00, 5.141665869436192970e+00, 4.671710754773619634e+00, 3.308906469280518792e+00, 2.260935542357682682e+00, 1.267027675320232083e+00, 5.310302445802553839e-01, 1.205186623337621693e-01])
# half_tweb = 1e-2 * np.array([3.929994704531686867e-01, 5.541758863744186137e-01, 6.039841197772257697e-01, 1.381458071395717635e+00, 1.123180483752061587e+00, 8.978255449258085719e-01, 7.057174470185272330e-01, 3.472195393658423224e-01, 1.716305123422086076e-01, 1.000000000000000056e-01,])
# wing_twist_coefficients = np.array([-1.864996758674682065e+00, -1.887563731814386303e+00])
# center_wing_half_span = np.array([3.000000000000000444e+00])
# transition_half_span = np.array([3.000000000000000444e+00])
# wing_half_span = np.array([1.771681807724542068e+01])
# center_wing_chord_stretch_coefficients = np.array([-5.574281883182186093e-01, -4.053422828244272580e+00, -3.341042257413223471e+00, -3.548134619452329108e+00])
# wing_root_chord = np.array([5.114889837458465927e+00])
# wing_tip_chord = np.array([1.694647233448457246e+00])
# wing_sweep = np.array([3.966501558870209276e+01])
# transition_sweep = np.array([3.143469885871966518e+01])
# oversized_payload_translation_x = np.array([9.999999999999998224e+00])
# oversized_payload_translation_z = np.array([0.000000000000000000e+00])
# oversized_payload_rotation = np.array([-3.514884289414112108e-14])
# cruise_trim_elevator_deflection = np.array([5.084434597044037440])

pitch_b = np.array([5.0])
half_ttop_b = 1e-2 * np.array([3.471430464523918569e-01, 4.710321324047168634e-01, 2.346720892339240194e+00, 5.141665869436192970e+00, 4.671710754773619634e+00, 3.308906469280518792e+00, 2.260935542357682682e+00, 1.267027675320232083e+00, 5.310302445802553839e-01, 1.205186623337621693e-01])
half_tweb_b = 1e-2 * np.array([3.929994704531686867e-01, 5.541758863744186137e-01, 6.039841197772257697e-01, 1.381458071395717635e+00, 1.123180483752061587e+00, 8.978255449258085719e-01, 7.057174470185272330e-01, 3.472195393658423224e-01, 1.716305123422086076e-01, 1.000000000000000056e-01,])
wing_twist_coefficients_b = np.array([-1.864996758674682065e+00, -1.887563731814386303e+00])
center_wing_half_span_b = np.array([3.000000000000000444e+00])
transition_half_span_b = np.array([3.000000000000000444e+00])
wing_half_span_b = np.array([1.771681807724542068e+01])
center_wing_chord_stretch_coefficients_b = np.array([-5.574281883182186093e-01, -4.053422828244272580e+00, -3.341042257413223471e+00, -3.548134619452329108e+00])
wing_root_chord_b = np.array([5.114889837458465927e+00])
wing_tip_chord_b = np.array([1.694647233448457246e+00])
wing_sweep_b = np.array([3.966501558870209276e+01])
transition_sweep_b = np.array([3.143469885871966518e+01])
oversized_payload_translation_x_b = np.array([9.999999999999998224e+00])
oversized_payload_translation_z_b = np.array([0.000000000000000000e+00])
oversized_payload_rotation_b = np.array([-3.514884289414112108e-14])
cruise_trim_elevator_deflection_b = np.array([5.084434597044037440])

x_star = np.concatenate([pitch_b,
                        half_ttop_a,# half_ttop_b,
                        half_tweb_a,# half_tweb_b,
                        wing_twist_coefficients_a,# wing_twist_coefficients_b,
                        center_wing_half_span_a,# center_wing_half_span_b,
                        transition_half_span_b,
                        wing_half_span_a,# wing_half_span_b,
                        center_wing_chord_stretch_coefficients_a,#center_wing_chord_stretch_coefficients_b,
                        wing_root_chord_b,
                        wing_tip_chord_b,
                        wing_sweep_b,
                        transition_sweep_b,
                        oversized_payload_translation_x_b,
                        oversized_payload_translation_z_b,
                        oversized_payload_rotation_b,
                        cruise_trim_elevator_deflection_b])


print('solution: ', x_star)


denominator = np.where(np.abs(x_star) > 1e-12, x_star, 1.0) # prevent divide-by-zero errors
error = [np.linalg.norm((x_history[i] - x_star) / denominator) for i in range(len(x_history))]


n = iteration_ids[-1] + 1


plt.figure(figsize=(4, 3))
# plt.plot(iteration_ids, error_history, label='error', linewidth=2)
plt.plot(iteration_ids, error, label='error', linewidth=2)
plt.grid(axis='y', color='lavender')
plt.xlabel('Inner-loop iteration')
plt.ylabel('Error')
plt.tight_layout()
# plt.savefig('examples/bwb/bwb_convergence.pdf', bbox_inches='tight')
# plt.savefig('examples/bwb/bwb_convergence.pdf')
plt.show()


# plt.figure(figsize=(4, 3))
# mu_hist = np.asarray(mu_history)
# for i in range(mu_hist.shape[1]):
#     plt.semilogy(iteration_ids, mu_hist[:, i], label=f'mu[{i}]', linewidth=2)
# plt.xlabel('Inner-loop iteration')
# plt.ylabel('Penalty parameters')
# plt.show()


plt.figure(figsize=(4, 3))
# plot the convergence of all variables in the x vector
x_history = np.array(x_history)
for i in range(x_history.shape[1]):
    # plt.plot(iteration_ids, x_history[:, i], label=f'x[{i}]', linewidth=2)
    # plt.semilogy(iteration_ids, x_history[:, i] + 10, label=f'x[{i}]', linewidth=2)

    plt.semilogy(iteration_ids, x_history[:, 0], linewidth=1.5) # pitch
    plt.semilogy(iteration_ids, x_history[:, 1:11], linewidth=1.5) # half_ttop
    plt.semilogy(iteration_ids, x_history[:, 11:21], linewidth=1.5) # half_tweb
    plt.semilogy(iteration_ids, abs(x_history[:, 21:23]), linewidth=1.5) # wing_twist_coefficients
    plt.semilogy(iteration_ids, x_history[:, 23:24], linewidth=1.5) # center_wing_half_span
    plt.semilogy(iteration_ids, x_history[:, 24:25], linewidth=1.5) # transition_half_span
    plt.semilogy(iteration_ids, x_history[:, 25:26], linewidth=1.5) # wing_half_span
    plt.semilogy(iteration_ids, abs(x_history[:, 26:30]), linewidth=1.5) # center_wing_chord_stretch_coefficients
    plt.semilogy(iteration_ids, x_history[:, 30:31], linewidth=1.5) # wing_root_chord
    plt.semilogy(iteration_ids, x_history[:, 31:32], linewidth=1.5) # wing_tip_chord
    plt.semilogy(iteration_ids, x_history[:, 32:33], linewidth=1.5) # wing_sweep
    plt.semilogy(iteration_ids, x_history[:, 33:34], linewidth=1.5) # transition_sweep
    plt.semilogy(iteration_ids, abs(x_history[:, 34:35]) + 1e-2, linewidth=1.5) # oversized_payload_translation_x
    plt.semilogy(iteration_ids, x_history[:, 35:36], linewidth=1.5) # oversized_payload_translation_z
    # plt.semilogy(iteration_ids, abs(x_history[:, 36:37]) + 1e-3, linewidth=1.5) # oversized_payload_rotation
    plt.semilogy(iteration_ids, x_history[:, 37:38], linewidth=1.5) # cruise_trim_elevator_deflection


plt.grid(axis='y', color='lavender')
plt.ylabel('Absolute variable values')
plt.xlabel('Inner-loop iteration')
plt.xlim(0, n - 1)
plt.tight_layout()
# plt.savefig('examples/bwb/bwb_variables.pdf', bbox_inches='tight')
# plt.savefig('examples/bwb/bwb_variables.pdf')
plt.show()
