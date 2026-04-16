from build_model_2 import build_model_2
from modopt import CSDLAlphaProblem
from modopt import PySLSQP
import cstate
import warnings
warnings.filterwarnings("ignore")
import gc

def subproblem_2(x, y, mu):
    print('SP2 x: ', x)
    sim_2, design_variables_2, additional_inputs_dict_SP2, additional_outputs_dict = build_model_2(x, y, mu)

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

    # # assign values to the sim variables
    # sim_2[additional_inputs_dict_SP2['pitch']] = pitch
    # sim_2[design_variables_2['half_ttop'].variable] = half_ttop
    # sim_2[design_variables_2['half_tweb'].variable] = half_tweb
    # sim_2[additional_inputs_dict_SP2['wing_twist_coefficients']] = wing_twist_coefficients
    # sim_2[additional_inputs_dict_SP2['center_wing_half_span']] = center_wing_half_span
    # sim_2[additional_inputs_dict_SP2['transition_half_span']] = transition_half_span
    # sim_2[additional_inputs_dict_SP2['wing_half_span']] = wing_half_span
    # sim_2[additional_inputs_dict_SP2['center_wing_chord_stretch_coefficients']] = center_wing_chord_stretch_coefficients
    # sim_2[additional_inputs_dict_SP2['wing_root_chord']] = wing_root_chord
    # sim_2[additional_inputs_dict_SP2['wing_tip_chord']] = wing_tip_chord
    # sim_2[additional_inputs_dict_SP2['wing_sweep']] = wing_sweep
    # sim_2[additional_inputs_dict_SP2['transition_sweep']] = transition_sweep
    # sim_2[design_variables_2['oversized_payload_translation_x'].variable] = oversized_payload_translation_x
    # sim_2[design_variables_2['oversized_payload_translation_z'].variable] = oversized_payload_translation_z
    # sim_2[design_variables_2['oversized_payload_rotation'].variable] = oversized_payload_rotation
    # sim_2[additional_inputs_dict_SP2['cruise_trim_elevator_deflection']] = cruise_trim_elevator_deflection

    # # assign y and mu and slack values to the sim
    # sim_2[additional_inputs_dict_SP2['y']] = y
    # sim_2[additional_inputs_dict_SP2['mu']] = mu

    # print('Checkpoint SP2!')

    prob = CSDLAlphaProblem(problem_name='SP2_V2', simulator=sim_2)
    optimizer = PySLSQP(prob, solver_options={'maxiter':300, 'acc':1e-5})
    optimizer.solve()
    # success = optimizer.results['success']
    # solution = optimizer.results['x']

    sim_2.run() # might be necessary to run the sim to update the cstate values after optimization

    cstate.cstate = sim_2[additional_outputs_dict['c']]
    print('SP2 cstate: ', sim_2[additional_outputs_dict['c']])

    solution = [
                pitch,
                sim_2[design_variables_2['half_ttop'].variable],
                sim_2[design_variables_2['half_tweb'].variable],
                wing_twist_coefficients,
                center_wing_half_span,
                transition_half_span,
                wing_half_span,
                center_wing_chord_stretch_coefficients,
                wing_root_chord,
                wing_tip_chord,
                wing_sweep,
                transition_sweep,
                sim_2[design_variables_2['oversized_payload_translation_x'].variable],
                sim_2[design_variables_2['oversized_payload_translation_z'].variable],
                sim_2[design_variables_2['oversized_payload_rotation'].variable],
                cruise_trim_elevator_deflection,
                ]
    
    print('SP2 solution: ', solution)

    gc.collect()

    return solution