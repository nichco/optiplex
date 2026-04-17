from build_model_1 import build_model_1
from modopt import CSDLAlphaProblem
from modopt import PySLSQP
import cstate
import warnings
warnings.filterwarnings("ignore")
import gc

def subproblem_1(x, y, mu):
    print('SP1 x: ', x)
    sim_1, design_variables_1, additional_inputs_dict_SP1, additional_outputs_dict = build_model_1(x, y, mu)

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
    # sim_1[design_variables_1['pitch'].variable] = pitch
    # sim_1[additional_inputs_dict_SP1['half_ttop']] = half_ttop
    # sim_1[additional_inputs_dict_SP1['half_tweb']] = half_tweb
    # sim_1[design_variables_1['wing_twist_coefficients'].variable] = wing_twist_coefficients
    # sim_1[design_variables_1['center_wing_half_span'].variable] = center_wing_half_span
    # sim_1[design_variables_1['transition_half_span'].variable] = transition_half_span
    # sim_1[design_variables_1['wing_half_span'].variable] = wing_half_span
    # sim_1[design_variables_1['center_wing_chord_stretch_coefficients'].variable] = center_wing_chord_stretch_coefficients
    # sim_1[design_variables_1['wing_root_chord'].variable] = wing_root_chord
    # sim_1[design_variables_1['wing_tip_chord'].variable] = wing_tip_chord
    # sim_1[design_variables_1['wing_sweep'].variable] = wing_sweep
    # sim_1[design_variables_1['transition_sweep'].variable] = transition_sweep
    # sim_1[additional_inputs_dict_SP1['oversized_payload_translation_x']] = oversized_payload_translation_x
    # sim_1[additional_inputs_dict_SP1['oversized_payload_translation_z']] = oversized_payload_translation_z
    # sim_1[additional_inputs_dict_SP1['oversized_payload_rotation']] = oversized_payload_rotation
    # sim_1[design_variables_1['cruise_trim_elevator_deflection'].variable] = cruise_trim_elevator_deflection

    # # assign y and mu and slack values to the sim
    # sim_1[additional_inputs_dict_SP1['y']] = y
    # sim_1[additional_inputs_dict_SP1['mu']] = mu

    # # sim_1.run()
    # print(sim_1[design_variables_1['wing_twist_coefficients'].variable])
    # print(wing_twist_coefficients)
    # print(sim_1[additional_inputs_dict_SP1['mu']])
    # # exit()

    # print('Checkpoint SP1!')

    prob = CSDLAlphaProblem(problem_name='SP1_V2', simulator=sim_1)
    # optimizer = PySLSQP(prob, solver_options={'maxiter':300, 'acc':1e-5}, readable_outputs=['x'])
    optimizer = PySLSQP(prob, solver_options={'maxiter':300, 'acc':1e-6}, readable_outputs=['x'])
    optimizer.solve()
    # success = optimizer.results['success']
    # solution = optimizer.results['x']

    sim_1.run() # might be necessary to run the sim to update the cstate values after optimization

    cstate.cstate = sim_1[additional_outputs_dict['c']]
    print('SP1 cstate: ', sim_1[additional_outputs_dict['c']])

    solution = [
                sim_1[design_variables_1['pitch'].variable],
                half_ttop,
                half_tweb,
                sim_1[design_variables_1['wing_twist_coefficients'].variable],
                sim_1[design_variables_1['center_wing_half_span'].variable],
                sim_1[design_variables_1['transition_half_span'].variable],
                sim_1[design_variables_1['wing_half_span'].variable],
                sim_1[design_variables_1['center_wing_chord_stretch_coefficients'].variable],
                sim_1[design_variables_1['wing_root_chord'].variable],
                sim_1[design_variables_1['wing_tip_chord'].variable],
                sim_1[design_variables_1['wing_sweep'].variable],
                sim_1[design_variables_1['transition_sweep'].variable],
                oversized_payload_translation_x,
                oversized_payload_translation_z,
                oversized_payload_rotation,
                sim_1[design_variables_1['cruise_trim_elevator_deflection'].variable],
                ]
    
    print('SP1 solution: ', solution)

    gc.collect()

    return solution