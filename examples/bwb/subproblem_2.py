from model_2 import sim, design_variables
from modopt import CSDLAlphaProblem
from modopt import PySLSQP


def subproblem_2(x, y, mu):

    # extract initial values from x
    pitch = x[0]
    half_ttop = x[1]
    half_tweb = x[2]
    wing_twist_coefficients = x[3]
    center_wing_half_span = x[4]
    transition_half_span = x[5]
    wing_half_span = x[6]
    center_wing_chord_stretch_coefficients = x[7]
    wing_root_chord = x[8]
    wing_tip_chord = x[9]
    wing_sweep = x[10]
    transition_sweep = x[11]
    oversized_payload_translation_x = x[12]
    oversized_payload_translation_z = x[13]
    oversized_payload_rotation = x[14]
    cruise_trim_elevator_deflection = x[15]

    # assign values to the sim
    sim[design_variables['pitch'].variable] = pitch
    sim[design_variables['half_ttop'].variable] = half_ttop
    sim[design_variables['half_tweb'].variable] = half_tweb
    sim[design_variables['wing_twist_coefficients'].variable] = wing_twist_coefficients
    sim[design_variables['center_wing_half_span'].variable] = center_wing_half_span
    sim[design_variables['transition_half_span'].variable] = transition_half_span
    sim[design_variables['wing_half_span'].variable] = wing_half_span
    sim[design_variables['center_wing_chord_stretch_coefficients'].variable] = center_wing_chord_stretch_coefficients
    sim[design_variables['wing_root_chord'].variable] = wing_root_chord
    sim[design_variables['wing_tip_chord'].variable] = wing_tip_chord
    sim[design_variables['wing_sweep'].variable] = wing_sweep
    sim[design_variables['transition_sweep'].variable] = transition_sweep
    sim[design_variables['oversized_payload_translation_x'].variable] = oversized_payload_translation_x
    sim[design_variables['oversized_payload_translation_z'].variable] = oversized_payload_translation_z
    sim[design_variables['oversized_payload_rotation'].variable] = oversized_payload_rotation
    sim[design_variables['cruise_trim_elevator_deflection'].variable] = cruise_trim_elevator_deflection


    # assign y and mu to the sim
    # sim[y_variable] = y
    # sim[mu_variable] = mu

    prob = CSDLAlphaProblem(simulator=sim)
    optimizer = PySLSQP(prob, solver_options={'maxiter':400, 'acc':1e-4}, readable_outputs=['x'])
    optimizer.solve()
    # success = optimizer.results['success']
    # solution = optimizer.results['x']


    solution = [sim[design_variables['pitch'].variable.value],
                sim[design_variables['half_ttop'].variable.value],
                sim[design_variables['half_tweb'].variable.value],
                sim[design_variables['wing_twist_coefficients'].variable.value],
                sim[design_variables['center_wing_half_span'].variable.value],
                sim[design_variables['transition_half_span'].variable.value],
                sim[design_variables['wing_half_span'].variable.value],
                sim[design_variables['center_wing_chord_stretch_coefficients'].variable.value],
                sim[design_variables['wing_root_chord'].variable.value],
                sim[design_variables['wing_tip_chord'].variable.value],
                sim[design_variables['wing_sweep'].variable.value],
                sim[design_variables['transition_sweep'].variable.value],
                sim[design_variables['oversized_payload_translation_x'].variable.value],
                sim[design_variables['oversized_payload_translation_z'].variable.value],
                sim[design_variables['oversized_payload_rotation'].variable.value],
                sim[design_variables['cruise_trim_elevator_deflection'].variable.value]]

    return solution