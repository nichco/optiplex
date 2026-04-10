import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import bsm3
import lsdo_geo
from VortexAD import PanelMethod
from VortexAD import find_cell_adjacency, TE_detection
from VortexAD import plot_pressure_distribution
import sys
import os
import meshio
import aframe as af
from aeroelastic_coupling_utils import NodalMap
from dataclasses import dataclass

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from geometry_functions import compute_volume
from geometry_functions import project_transition_volume_points
from geometry_functions_newer import initial_refit, setup_and_evaluate_geometry_parameterization
import pickle
from additional_solvers import compute_static_margin, estimate_CDw, estimate_sectional_CDw, estimate_Cf
from additional_solvers import estimate_fuel_burn, atmos_model

lfs.num_workers=1
save_meshes = True

print('Building model 1 simulator...')

recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()

# region ============================ geometry import and setup ============================
file_name = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
file_path = 'examples/bwb/geometry/' 

# importing geometry
# imported_geometry = lsdo_geo.import_geometry(
#     file_path + file_name,
#     parallelize=True,
# )
imported_function_set = lfs.import_file_patched(file_name=file_path + file_name, parallelize=False)
imported_geometry = lsdo_geo.Geometry(functions=imported_function_set.functions, 
                                      function_names=imported_function_set.function_names,
                                      name='imported_geometry',
                                      space=imported_function_set.space)

# imported_geometry.plot(camera=camera)

geometry, lower_discretizations, upper_discretizations, \
    lower_parametric_coordinates, upper_parametric_coordinates, \
    lower_inverted_fitting_matrix, upper_inverted_fitting_matrix, \
    spanwise_section_boundaries = initial_refit(imported_geometry, plot=False)

ft2m = 1/3.28084
oversized_payload_length = 33*ft2m
oversized_payload_width = 13*ft2m
oversized_payload_height = 10*ft2m
oversized_payload_corners = np.array([
    [[[0., -oversized_payload_width/2, -oversized_payload_height/2],
    [0., -oversized_payload_width/2, oversized_payload_height/2]],
    [[0., oversized_payload_width/2, -oversized_payload_height/2],
    [0., oversized_payload_width/2, oversized_payload_height/2]]],
    [[[oversized_payload_length, -oversized_payload_width/2, -oversized_payload_height/2],
    [oversized_payload_length, -oversized_payload_width/2, oversized_payload_height/2]],
    [[oversized_payload_length, oversized_payload_width/2, -oversized_payload_height/2],
    [oversized_payload_length, oversized_payload_width/2, oversized_payload_height/2]]],
])
payload_face_function_space = lfs.BSplineSpaceNew(num_parametric_dimensions=2, degree=(1, 1), coefficients_shape=(2, 2))
oversized_payload_u0_face = lfs.Function(space=payload_face_function_space, coefficients=oversized_payload_corners[0, :, :, :].reshape((2,2,3)), name='oversized_payload_u0_face')
oversized_payload_u1_face = lfs.Function(space=payload_face_function_space, coefficients=oversized_payload_corners[1, :, :, :].reshape((2,2,3)), name='oversized_payload_u1_face')
oversized_payload_v0_face = lfs.Function(space=payload_face_function_space, coefficients=oversized_payload_corners[:, 0, :, :].reshape((2,2,3)), name='oversized_payload_v0_face')
oversized_payload_v1_face = lfs.Function(space=payload_face_function_space, coefficients=oversized_payload_corners[:, 1, :, :].reshape((2,2,3)), name='oversized_payload_v1_face')
oversized_payload_w0_face = lfs.Function(space=payload_face_function_space, coefficients=oversized_payload_corners[:, :, 0, :].reshape((2,2,3)), name='oversized_payload_w0_face')
oversized_payload_w1_face = lfs.Function(space=payload_face_function_space, coefficients=oversized_payload_corners[:, :, 1, :].reshape((2,2,3)), name='oversized_payload_w1_face')
function_dict = {0: oversized_payload_u0_face, 1: oversized_payload_u1_face, 2: oversized_payload_v0_face, 3: oversized_payload_v1_face, 4: oversized_payload_w0_face, 5: oversized_payload_w1_face}
oversized_payload = lsdo_geo.Geometry(functions=function_dict, name='oversized_payload')
initial_translation = np.array([6., 0., 0.])
# initial_translation = np.array([10., 0., 0.])
# oversized_payload_center = np.array([oversized_payload_length/2, 0., 0.])
# oversized_payload.rotate(rotation_origin=oversized_payload_center, axis_vector=np.array([0., 0., 1.]), angles=np.pi/2)
oversized_payload.translate(initial_translation)

in2m = 1/39.3701
num_rows = 6
num_cols = 3
payload_package_1_length = 88*num_rows*in2m
payload_package_1_width = 108*num_cols*in2m
payload_package_1_height = 96*in2m
payload_package_1_corners = np.array([
    [[[0., -payload_package_1_width/2, -payload_package_1_height/2],
    [0., -payload_package_1_width/2, payload_package_1_height/2]],
    [[0., payload_package_1_width/2, -payload_package_1_height/2],
    [0., payload_package_1_width/2, payload_package_1_height/2]]],
    [[[payload_package_1_length, -payload_package_1_width/2, -payload_package_1_height/2],
    [payload_package_1_length, -payload_package_1_width/2, payload_package_1_height/2]],
    [[payload_package_1_length, payload_package_1_width/2, -payload_package_1_height/2],
    [payload_package_1_length, payload_package_1_width/2, payload_package_1_height/2]]],
])
payload_package_1_u0_face = lfs.Function(space=payload_face_function_space, coefficients=payload_package_1_corners[0, :, :, :].reshape((2,2,3)), name='payload_package_1_u0_face')
payload_package_1_u1_face = lfs.Function(space=payload_face_function_space, coefficients=payload_package_1_corners[1, :, :, :].reshape((2,2,3)), name='payload_package_1_u1_face')
payload_package_1_v0_face = lfs.Function(space=payload_face_function_space, coefficients=payload_package_1_corners[:, 0, :, :].reshape((2,2,3)), name='payload_package_1_v0_face')
payload_package_1_v1_face = lfs.Function(space=payload_face_function_space, coefficients=payload_package_1_corners[:, 1, :, :].reshape((2,2,3)), name='payload_package_1_v1_face')
payload_package_1_w0_face = lfs.Function(space=payload_face_function_space, coefficients=payload_package_1_corners[:, :, 0, :].reshape((2,2,3)), name='payload_package_1_w0_face')
payload_package_1_w1_face = lfs.Function(space=payload_face_function_space, coefficients=payload_package_1_corners[:, :, 1, :].reshape((2,2,3)), name='payload_package_1_w1_face')
function_dict = {0: payload_package_1_u0_face, 1: payload_package_1_u1_face, 2: payload_package_1_v0_face, 3: payload_package_1_v1_face, 4: payload_package_1_w0_face, 5: payload_package_1_w1_face}
payload_package_1 = lsdo_geo.Geometry(functions=function_dict, name='payload_package_1')
initial_translation = np.array([8., 0., 0.])
payload_package_1.translate(initial_translation)


# geometry_plot = geometry.plot(opacity=0.5, show=False)W
# # payload_package_plot = payload_package_1.plot(additional_plotting_elements=[geometry_plot], color='#F3E500', opacity=0.9, show=False)
# # oversized_payload.plot(additional_plotting_elements=[payload_package_plot], interactive=True, color='#FC8900', opacity=0.9)
# oversized_payload.plot(additional_plotting_elements=[geometry_plot], interactive=True, color='#FC8900', opacity=0.9)
# exit()
# endregion

# region ============================ aero mesh setup ============================
file_name = 'bwbv2_no_wingtip_coarse_refined_flat_5_9_17_5264_cap_tess_3_LE_bunch_quad.msh'
mesh = meshio.read(file_path + file_name)

points_orig = mesh.points
cells = mesh.cells
cells_dict = mesh.cells_dict

cell_adjacency_data = find_cell_adjacency(points=points_orig, cells=cells_dict)

points_orig = cell_adjacency_data[0] 
cells_dict = cell_adjacency_data[1] 
cell_adjacency = cell_adjacency_data[2] 
edges2cells = cell_adjacency_data[3]
points2cells = cell_adjacency_data[4]

TE_properties = TE_detection(points=points_orig,
                             cells=cells_dict,
                             edges2cells=edges2cells,
                             threshold_theta=125.
                             )

upper_TE_cells = TE_properties[0] 
lower_TE_cells = TE_properties[1] 
TE_edges = TE_properties[2] 
TE_node_indices = TE_properties[3]

cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types: combined_cells += cells_dict[cell_type].tolist()

if False:
    TE_coloring = np.zeros(shape=(len(combined_cells),))
    TE_coloring[upper_TE_cells] = 1
    TE_coloring[lower_TE_cells] = -1

    plot_pressure_distribution(points_orig, TE_coloring, connectivity=combined_cells, interactive=True, top_view=False, cmap='rainbow')
    exit()

projected_panel_mesh = geometry.project(points_orig, 
                                        grid_search_density_parameter=2, 
                                        newton_tolerance=1.e-5, 
                                        grid_search_density_cutoff=7,
                                        projection_tolerance=1.e-2,
                                        force_reprojection=False, 
                                        plot=False
                                        )

# project panel centers
cell_types = cells_dict.keys()
combined_cells = []
for cell_type in cell_types: combined_cells += cells_dict[cell_type].tolist()

if save_meshes:
    print('projecting panel centers')
    panel_centers = np.zeros((len(combined_cells), 3))
    for i, cell in enumerate(combined_cells):
        panel_centers[i] = np.mean(points_orig[cell], axis=0)

    panel_centers = geometry.project(panel_centers, 
                                grid_search_density_parameter=2, 
                                newton_tolerance=1.e-5,
                                grid_search_density_cutoff=7,
                                projection_tolerance=1.e-2,
                                force_reprojection=False, 
                                plot=False,
                                )
    with open('mesh_projections/panel_centers.pkl', 'wb') as f:
        pickle.dump(panel_centers, f)
    print('done projecting panel centers')


# load the projected panel centers from a file
with open('mesh_projections/panel_centers.pkl', 'rb') as f:
    projected_panel_centers = pickle.load(f)
# endregion

# region ============================ structures mesh setup ============================
num_nonwing_nodes = 4
num_wing_nodes = 7

# LE and TE points
LE_nonwing_pts = np.array([[0., 0., 0.],
                           [3.899, 2.5, 0.],
                           [9.813, 5.0, 0.],
                           [15.541, 7.934, 0.659]])

TE_nonwing_pts = np.array([[30., 0., 0.],
                           [30., 2.5, 0.],
                           [30., 5.0, 0.],
                           [24.88, 7.934, 0.691]])

wing_root_LE_TE = [np.array([17.815, 9.891, 1.040]), # LE
                   np.array([23.815, 9.891, 1.040]), # TE
                   ]

wing_tip_LE_TE = [np.array([29.019, 25.852, 2.123]), # LE
                  np.array([32.016, 25.852, 2.254]), # TE
                  ]

LE_wing_pts = np.linspace(wing_root_LE_TE[0], wing_tip_LE_TE[0], num_wing_nodes)
TE_wing_pts = np.linspace(wing_root_LE_TE[1], wing_tip_LE_TE[1], num_wing_nodes)

LE_points = np.concatenate((LE_nonwing_pts, LE_wing_pts)) # we don't want the center line
TE_points = np.concatenate((TE_nonwing_pts, TE_wing_pts)) # we don't want the center line

LE_points_projected = geometry.project(LE_points, plot=False)
TE_points_projected = geometry.project(TE_points, plot=False)
                                                                               
# airfoil top and bottom points
top_nonwing_pts = np.array([
    [8.973, 0, 2.251],
    [11.706, 2.5, 1.958],
    [15.851, 5., 1.514],
    [18.352, 7.934, 1.415],
])

bottom_nonwing_pts = np.array([
    [8.973, 0, -2.251],
    [11.706, 2.5, -1.958],
    [15.851, 5., -1.514],
    [18.398, 7.934, 0.078],
])

wing_root_top_bottom = [
    np.array([20.001, 9.891, 1.537]), # top
    np.array([20.040, 9.891, 0.699]), # bottom
]

tip_top_bottom = [
    np.array([30.102, 25.852, 2.384]), # top
    np.array([30.137, 25.852, 2.026]), # bottom
]

top_wing_pts = np.linspace(wing_root_top_bottom[0], tip_top_bottom[0], num_wing_nodes)
bottom_wing_pts = np.linspace(wing_root_top_bottom[1], tip_top_bottom[1], num_wing_nodes)

top_pts = np.concatenate((top_nonwing_pts, top_wing_pts))[1:,:] # we don't want the center line
bot_pts = np.concatenate((bottom_nonwing_pts, bottom_wing_pts))[1:,:] # we don't want the center line

top_pts_projected = geometry.project(top_pts, plot=False)
bot_pts_projected = geometry.project(bot_pts, plot=False)
# endregion

# region ============================ mass properties setup and point projections ============================
lbf_to_N = 4.44822
# mass properties setup
empty_weight_lbf = 126636 #lbf
empty_weight_N = empty_weight_lbf * lbf_to_N

# a guess on initial wing mass (acts as a deficit )
wing_mass_0 = 5000 # kg
wing_weight_0 = wing_mass_0 * 9.81

empty_weight_N_nowing = empty_weight_N - wing_weight_0

# mass point projections
engine_sec_LE_0 = np.array([7.347, -4., 0.])
engine_sec_TE_0 = np.array([30., -4., 0.])
engine_loc_0 = np.array([24.702, -4., 0.824]) # 80% of center span, around 80% of chord
ecf_0 = (engine_loc_0[0]-engine_sec_LE_0[0])/(engine_sec_TE_0[0]-engine_sec_LE_0[0]) # engine chord fraction
# ecf_0 here is around 76.6% of the chord (just a starting point)
# ecf_0 = 0.8 # engine chord fraction


engine_loc_parametric = geometry.project(engine_loc_0, plot=False)
engine_sec_LE_parametric = geometry.project(engine_sec_LE_0, plot=False)
engine_sec_TE_parametric = geometry.project(engine_sec_TE_0, plot=False)


transition_volume_LE_points = np.array([
    [9.813, 5.0, 0.],
    [12.311, 6.223, 0.158],
    [14.707, 7.445, 0.510],
    [16.604, 8.668, 0.853],
    [17.815, 9.891, 1.040],
])

transition_volume_TE_points = np.array([
    [30., 5.0, 0.],
    [28.627, 6.223, 0.167],
    [25.813, 7.445, 0.545],
    [23.963, 8.668, 0.870],
    [23.815, 9.891, 1.040],
])

transition_volume_LE_points_proj = geometry.project(transition_volume_LE_points)
transition_volume_TE_points_proj = geometry.project(transition_volume_TE_points)

transition_volume_projection_points = project_transition_volume_points(
    geometry, 
    transition_volume_LE_points, 
    transition_volume_TE_points
)

# endregion

# region ============================ mission parameters ===========================
nmi_to_m = 1852.
payload_weight_lbf = np.array([160000])
cruise_range_nmi = 2000
payload_weight_N = payload_weight_lbf*lbf_to_N
cruise_range_m = cruise_range_nmi*nmi_to_m
num_nodes = 2 # cruise plus 1 perturbation for stability analysis
dalpha_stab = 0.1 # for stability analysis
cruise_mach = 0.7
# cruise_mach = 0.75
cruise_h = 30000 # altitude in feet
cruise_h_km = cruise_h*0.3048/1000

'''
NOTE:
- assume cruise uses 75% of fuel fraction (to be conservative)
- original fuel weight: 44885.25 lbf (according to Nick)
'''
# endregion

# region ============================ geometry parameterization ===========================
lower_oml = geometry.functions[0]
upper_oml = geometry.functions[1]

upper_wing_root = upper_oml.evaluate(np.array([[0.667, spanwise_section_boundaries[-2]]]), plot=False).flatten()
lower_wing_root = lower_oml.evaluate(np.array([[0.667, spanwise_section_boundaries[-2]]]), plot=False).flatten()
wing_root = (upper_wing_root + lower_wing_root) / 2
upper_wing_tip = upper_oml.evaluate(np.array([[0.667, spanwise_section_boundaries[-1]]]), plot=False).flatten()
lower_wing_tip = lower_oml.evaluate(np.array([[0.667, spanwise_section_boundaries[-1]]]), plot=False).flatten()
wing_tip = (upper_wing_tip + lower_wing_tip) / 2
upper_center_wing_root = upper_oml.evaluate(np.array([[0.667, 0.5]]), plot=False).flatten()
lower_center_wing_root = lower_oml.evaluate(np.array([[0.667, 0.5]]), plot=False).flatten()
center_wing_root = (upper_center_wing_root + lower_center_wing_root) / 2
upper_center_wing_tip = upper_oml.evaluate(np.array([[0.667, spanwise_section_boundaries[-3]]]), plot=False).flatten()
lower_center_wing_tip = lower_oml.evaluate(np.array([[0.667, spanwise_section_boundaries[-3]]]), plot=False).flatten()
center_wing_tip = (upper_center_wing_tip + lower_center_wing_tip) / 2

wing_root_leading_edge = upper_oml.evaluate(np.array([[0.0, spanwise_section_boundaries[-2]]]), plot=False).flatten()
wing_root_trailing_edge = upper_oml.evaluate(np.array([[1.0, spanwise_section_boundaries[-2]]]), plot=False).flatten()
wing_tip_leading_edge = upper_oml.evaluate(np.array([[0.0, spanwise_section_boundaries[-1]]]), plot=False).flatten()
wing_tip_trailing_edge = upper_oml.evaluate(np.array([[1.0, spanwise_section_boundaries[-1]]]), plot=False).flatten()
center_wing_root_leading_edge = upper_oml.evaluate(np.array([[0.0, 0.5]]), plot=False).flatten()
center_wing_root_trailing_edge = upper_oml.evaluate(np.array([[1.0, 0.5]]), plot=False).flatten()
center_wing_tip_leading_edge = upper_oml.evaluate(np.array([[0.0, spanwise_section_boundaries[-3]]]), plot=False).flatten()
center_wing_tip_trailing_edge = upper_oml.evaluate(np.array([[1.0, spanwise_section_boundaries[-3]]]), plot=False).flatten()
center_wing_station_2_leading_edge = upper_oml.evaluate(np.array([[0.0, 0.5 + (spanwise_section_boundaries[-3]-0.5)/3]]), plot=False).flatten()
center_wing_station_2_trailing_edge = upper_oml.evaluate(np.array([[1.0, 0.5 + (spanwise_section_boundaries[-3]-0.5)/3]]), plot=False).flatten()
center_wing_station_3_leading_edge = upper_oml.evaluate(np.array([[0.0, 0.5 + 2*(spanwise_section_boundaries[-3]-0.5)/3]]), plot=False).flatten()
center_wing_station_3_trailing_edge = upper_oml.evaluate(np.array([[1.0, 0.5 + 2*(spanwise_section_boundaries[-3]-0.5)/3]]), plot=False).flatten()

wing_root_to_tip_vector = wing_tip - wing_root
center_wing_root_to_tip_vector = center_wing_tip - center_wing_root
transition_root_to_tip_vector = wing_root - center_wing_tip

wing_sweep_computed = csdl.arctan2(wing_root_to_tip_vector[0], wing_root_to_tip_vector[1])
transition_sweep_computed = csdl.arctan2(transition_root_to_tip_vector[0], transition_root_to_tip_vector[1])
# center_wing_sweep_computed = csdl.arctan2(center_wing_root_to_tip_vector[0], center_wing_root_to_tip_vector[1])

wing_half_span_computed = wing_root_to_tip_vector[1]
center_wing_half_span_computed = center_wing_root_to_tip_vector[1]
transition_half_span_computed = transition_root_to_tip_vector[1]

wing_dihedral_computed = csdl.arctan2(wing_root_to_tip_vector[2], wing_root_to_tip_vector[1])
center_wing_dihedral_computed = csdl.arctan2(center_wing_root_to_tip_vector[2], center_wing_root_to_tip_vector[1])
transition_dihedral_computed = csdl.arctan2(transition_root_to_tip_vector[2], transition_root_to_tip_vector[1])

wing_root_chord_computed = wing_root_trailing_edge[0] - wing_root_leading_edge[0]
wing_tip_chord_computed = wing_tip_trailing_edge[0] - wing_tip_leading_edge[0]
center_wing_root_chord_computed = center_wing_root_trailing_edge[0] - center_wing_root_leading_edge[0]
center_wing_tip_chord_computed = center_wing_tip_trailing_edge[0] - center_wing_tip_leading_edge[0]
center_wing_station_2_chord_computed = center_wing_station_2_trailing_edge[0] - center_wing_station_2_leading_edge[0]
center_wing_station_3_chord_computed = center_wing_station_3_trailing_edge[0] - center_wing_station_3_leading_edge[0]
center_wing_chords_computed = csdl.concatenate((center_wing_root_chord_computed.reshape(1,), 
                                                center_wing_station_2_chord_computed.reshape(1,), 
                                                center_wing_station_3_chord_computed.reshape(1,), 
                                                center_wing_tip_chord_computed.reshape(1,)), axis=0)

wing_root_thickness_computed = upper_wing_root[2] - lower_wing_root[2]
wing_tip_thickness_computed = upper_wing_tip[2] - lower_wing_tip[2]
center_wing_root_thickness_computed = upper_center_wing_root[2] - lower_center_wing_root[2]
center_wing_tip_thickness_computed = upper_center_wing_tip[2] - lower_center_wing_tip[2]

num_wing_twist_design_variables = 2
num_center_wing_spanwise_design_dofs = 4
wing_sweep = csdl.Variable(value=wing_sweep_computed.value*180/np.pi) # convert to degrees for design variable
transition_sweep = csdl.Variable(value=transition_sweep_computed.value*180/np.pi) # convert to degrees for design variable
wing_half_span = csdl.Variable(value=wing_half_span_computed.value)
center_wing_half_span = csdl.Variable(value=center_wing_half_span_computed.value)
transition_half_span = csdl.Variable(value=transition_half_span_computed.value)
wing_dihedral = csdl.Variable(value=wing_dihedral_computed.value)
center_wing_dihedral = csdl.Variable(value=center_wing_dihedral_computed.value)
transition_dihedral = csdl.Variable(value=transition_dihedral_computed.value)
# wing_root_chord = csdl.Variable(value=wing_root_chord_computed.value)
wing_root_chord = csdl.Variable(value=wing_root_chord_computed.value)
wing_tip_chord = csdl.Variable(value=wing_tip_chord_computed.value)
# center_wing_root_chord = csdl.Variable(value=center_wing_root_chord_computed.value)
# center_wing_tip_chord = csdl.Variable(value=center_wing_tip_chord_computed.value)
# center_wing_chords = csdl.Variable(shape=(num_center_wing_spanwise_design_dofs,), value=center_wing_chords_computed.value)
# center_wing_chords = csdl.Variable(shape=(center_wing_spanwise_design_dofs,), value=center_wing_chords_computed.value * np.array([1., 1., 1., 0.8]))
center_wing_chord_stretch_coefficients = csdl.Variable(shape=(num_center_wing_spanwise_design_dofs,), value=0.)
wing_root_thickness = csdl.Variable(value=wing_root_thickness_computed.value)
wing_tip_thickness = csdl.Variable(value=wing_tip_thickness_computed.value)
center_wing_root_thickness = csdl.Variable(value=center_wing_root_thickness_computed.value)
center_wing_tip_thickness = csdl.Variable(value=center_wing_tip_thickness_computed.value)

wing_twist_coefficients = csdl.Variable(shape=(num_wing_twist_design_variables,), value=0.)

center_wing_twist_dv = csdl.Variable(shape=(num_center_wing_spanwise_design_dofs-1,), value=0.)
# center_wing_twist_coefficients = csdl.Variable(shape=(num_center_wing_spanwise_design_dofs,), value=0.)
center_wing_twist_coefficients = csdl.concatenate((csdl.Variable(value=0.), center_wing_twist_dv), axis=0)

cruise_trim_elevator_deflection = csdl.Variable(value=0.)

geometric_variables = {
    'wing_sweep': wing_sweep,
    'transition_sweep': transition_sweep,
    'wing_half_span': wing_half_span,
    'center_wing_half_span': center_wing_half_span,
    'transition_half_span': transition_half_span,
    # 'wing_dihedral': wing_dihedral,
    # 'center_wing_dihedral': center_wing_dihedral,
    # 'transition_dihedral': transition_dihedral,
    'wing_root_chord': wing_root_chord,
    'wing_tip_chord': wing_tip_chord,
    # 'center_wing_chords': center_wing_chords,
    'center_wing_chord_stretch_coefficients': center_wing_chord_stretch_coefficients,
    # 'wing_root_thickness': wing_root_thickness,
    # 'wing_tip_thickness': wing_tip_thickness,
    # 'center_wing_root_thickness': center_wing_root_thickness,
    # 'center_wing_tip_thickness': center_wing_tip_thickness,
    'wing_twist_coefficients': wing_twist_coefficients,
    # 'center_wing_twist_coefficients': center_wing_twist_coefficients,
    'elevator_deflection': cruise_trim_elevator_deflection
    }

geometry = setup_and_evaluate_geometry_parameterization(imported_geometry, geometric_variables,
                                                        lower_discretizations, upper_discretizations, 
                                                        lower_parametric_coordinates, upper_parametric_coordinates,
                                                        lower_inverted_fitting_matrix, upper_inverted_fitting_matrix,
                                                        spanwise_section_boundaries, plot=False)

# region payload rigid body parameterization
oversized_payload_translation_x = csdl.Variable(value=0.)
oversized_payload_translation_z = csdl.Variable(value=0.)
oversized_payload_translation = csdl.concatenate((oversized_payload_translation_x.reshape(1,), csdl.Variable(value=0.).reshape(1,), oversized_payload_translation_z.reshape(1,)), axis=0)
oversized_payload_rotation = csdl.Variable(value=0.) # only allow pitch rotation since it should be symmetric and flat

oversized_payload_center = np.array([oversized_payload_length/2, 0., 0.]) + initial_translation
oversized_payload.rotate(rotation_origin=oversized_payload_center, axis_vector=np.array([0., 1., 0.]), angles=oversized_payload_rotation, units='degrees')
oversized_payload.translate(oversized_payload_translation)

# endregion payload rigid body parameterization
# endregion

# region ============================ geometry representations ===========================
spanwise_resolution = 50
leading_edge_parametric_coordinates = []
trailing_edge_parametric_coordinates = []
for i in range(spanwise_resolution):
    leading_edge_parametric_coordinates.append((0, np.array([0., i/(spanwise_resolution-1)])))
    trailing_edge_parametric_coordinates.append((1, np.array([1., i/(spanwise_resolution-1)])))

leading_edge_curve = geometry.evaluate(leading_edge_parametric_coordinates, plot=False)
trailing_edge_curve = geometry.evaluate(trailing_edge_parametric_coordinates, plot=False)
section_spans = leading_edge_curve[1:,1] - leading_edge_curve[:-1,1]
section_chords = (trailing_edge_curve[1:,0]+trailing_edge_curve[0:-1,0])/2 - (leading_edge_curve[1:,0]+leading_edge_curve[:-1,0])/2
planform_area_strips = section_chords*section_spans
planform_area = csdl.sum(planform_area_strips)
# print(f'planform area: {planform_area.value}')
MAC_integral_term = section_spans * section_chords**2 / 2
MAC = csdl.sum(MAC_integral_term) * 2 / planform_area
# print(f'MAC: {MAC.value}')

parametric_max_t_c_location = 0.667
upper_skin_max_thickness_locations_parametric_u = parametric_max_t_c_location*np.ones(shape=(spanwise_resolution,))
upper_skin_max_thickness_locations_parametric_v = np.linspace(0, 1, spanwise_resolution)
upper_skin_max_thickness_locations_parametric = np.stack([upper_skin_max_thickness_locations_parametric_u, upper_skin_max_thickness_locations_parametric_v], axis=1)
upper_skin_max_thickness_locations = upper_oml.evaluate(upper_skin_max_thickness_locations_parametric, plot=False)
lower_skin_max_thickness_locations_parametric_u = parametric_max_t_c_location*np.ones(shape=(spanwise_resolution,))
lower_skin_max_thickness_locations_parametric_v = np.linspace(0, 1, spanwise_resolution)
lower_skin_max_thickness_locations_parametric = np.stack([lower_skin_max_thickness_locations_parametric_u, lower_skin_max_thickness_locations_parametric_v], axis=1)
lower_skin_max_thickness_locations = lower_oml.evaluate(lower_skin_max_thickness_locations_parametric, plot=False)
thicknesses_at_max_locations = upper_skin_max_thickness_locations[:,2] - lower_skin_max_thickness_locations[:,2]
thicknesses_at_max_locations = (thicknesses_at_max_locations[1:] + thicknesses_at_max_locations[:-1]) / 2
# thicknesses_at_max_locations = thicknesses_at_max_locations*1.1 # This is just cause I'm definitely not getting the perfect max thickness location
section_t_c = thicknesses_at_max_locations / section_chords

quarter_chord_for_each_section = 0.75*leading_edge_curve + 0.25*trailing_edge_curve
sweep_vectors = quarter_chord_for_each_section[1:,:] - quarter_chord_for_each_section[:-1,:]
section_sweeps = csdl.absolute(csdl.arctan2(sweep_vectors[:,0], sweep_vectors[:,1]))

chordwise_resolution = 100
lower_oml = geometry.functions[0]
upper_oml = geometry.functions[1]
lower_discretization_parametric_grid = lower_oml.space.generate_parametric_grid((chordwise_resolution, spanwise_resolution))
upper_discretization_parametric_grid = upper_oml.space.generate_parametric_grid((chordwise_resolution, spanwise_resolution))
lower_oml_discretization_points = lower_oml.evaluate(lower_discretization_parametric_grid, plot=False).reshape((chordwise_resolution, spanwise_resolution, 3))
upper_oml_discretization_points = upper_oml.evaluate(upper_discretization_parametric_grid, plot=False).reshape((chordwise_resolution, spanwise_resolution, 3))
# lower_oml_u_vectors = lower_oml_discretization_points[1:,:,:] - lower_oml_discretization_points[:-1,:,:]
# lower_oml_v_vectors = lower_oml_discretization_points[:,1:,:] - lower_oml_discretization_points[:,:-1,:]
# upper_oml_u_vectors = upper_oml_discretization_points[1:,:,:] - upper_oml_discretization_points[:-1,:,:]
# upper_oml_v_vectors = upper_oml_discretization_points[:,1:,:] - upper_oml_discretization_points[:,:-1,:]
# lower_oml_area = csdl.norm(csdl.cross(lower_oml_u_vectors[:,1:], lower_oml_v_vectors[1:,:]), axes=3)
# upper_oml_area = csdl.norm(csdl.cross(upper_oml_u_vectors[:,1:], upper_oml_v_vectors[1:,:]), axes=3)
# wetted_area = csdl.sum(lower_oml_area) + csdl.sum(upper_oml_area)
# print(f'wetted area: {wetted_area.value}')


# volume_mesh = csdl.Variable(shape=(chordwise_resolution, spanwise_resolution, 2, 3), value=0.)
# volume_mesh = volume_mesh.set(csdl.slice[:,:,0,:], lower_oml_discretization_points)
# volume_mesh = volume_mesh.set(csdl.slice[:,:,1,:], upper_oml_discretization_points)
# volume_u_vectors = volume_mesh[1:,:,:,:] - volume_mesh[:-1,:,:,:]
# volume_v_vectors = volume_mesh[:,1:,:,:] - volume_mesh[:,:-1,:,:]
# volume_w_vectors = volume_mesh[:,:,1:,:] - volume_mesh[:,:,0:-1,:]
# volume_components = csdl.einsum(csdl.cross(volume_u_vectors[:,1:,0,:], volume_v_vectors[1:,:,0,:], axis=2), volume_w_vectors[1:,1:,0,:], action='ijk,ijk->ij')
# volume = csdl.sum(volume_components)
# print(volume.value)


# region geometric non-interference constraints setup
projection_model = bsm3.FunctionSetProjectionModel(
    function_set=geometry,
    warm_start_nu=50, # controls per-patch triangulation resolution
    warm_start_nv=50,
    sdf=True,
)

oversized_payload_discretization_parametric_coordinates = oversized_payload.generate_parametric_grid(grid_resolution=3)
oversized_payload_discretization_parametric_coordinates = [
    parametric_coordinate
    for parametric_coordinate in oversized_payload_discretization_parametric_coordinates
    if 0 not in parametric_coordinate[1] and 1 not in parametric_coordinate[1] \
        or parametric_coordinate[0] == 0 \
        or parametric_coordinate[0] == 1 \
        or (parametric_coordinate[0] == 2 or parametric_coordinate[0] == 3) and (parametric_coordinate[1][1] == 0 or parametric_coordinate[1][1] == 1) and (parametric_coordinate[1][0] != 0 and parametric_coordinate[1][0] != 1)
]

oversized_payload_discretization = oversized_payload.evaluate(oversized_payload_discretization_parametric_coordinates, plot=False)

geometry_coefficients_stacked = geometry.stack_coefficients()

sdf_op = bsm3.FunctionSetClosestDistanceOperation(model=projection_model)
oversized_payload_sdf_values = sdf_op.evaluate(coefficients=geometry_coefficients_stacked, points=oversized_payload_discretization)

oversized_payload_u0_end = oversized_payload.evaluate([(0, np.array([[0.5, 0.5]]))]).flatten()
oversized_payload_u1_end = oversized_payload.evaluate([(1, np.array([[0.5, 0.5]]))]).flatten()
oversized_payload_center_of_mass = (oversized_payload_u0_end + oversized_payload_u1_end) / 2
# endregion geometric non-interference constraints setup

# endregion

recorder.inline = False

# region ============================ aero solver setup ============================
panel_mesh = geometry.evaluate(projected_panel_mesh, plot=False)
panel_mesh.add_name('panel_mesh')
panel_mesh = panel_mesh.expand((1,) + panel_mesh.shape, 'ij->aij')
panel_mesh.save()
panel_center = geometry.evaluate(projected_panel_centers, plot=False)

# plotting mesh for debugging the geometry --> need recorder.inline = True
if False:
    TE_coloring = np.zeros(shape=(len(combined_cells),))
    TE_coloring[upper_TE_cells] = 1
    TE_coloring[lower_TE_cells] = -1

    plot_pressure_distribution(panel_mesh.value, TE_coloring, connectivity=combined_cells, interactive=True, top_view=False, cmap='rainbow')
    exit()

pitch = csdl.Variable(value=5) # DV

# atmosphere model
rho_c, a_c, mu_c = atmos_model(cruise_h_km) # properties at cruise altitude

V_cruise = a_c*cruise_mach
V_cruise_array = csdl.Variable(value=0)
V_cruise_array = V_cruise_array.set(csdl.slice[:], V_cruise)

V_inf_array = csdl.Variable(value=np.zeros((num_nodes,)))
V_inf_array = V_inf_array.set(csdl.slice[0], V_cruise_array) # cruise speeds
V_inf_array = V_inf_array.set(csdl.slice[1], value=V_cruise) # perturbation for static margin at same altitude

V_vec = csdl.Variable(value=0., shape=(num_nodes,3))
V_vec = V_vec.set(csdl.slice[:,0], value=-V_inf_array)

rho_array = csdl.Variable(value=np.zeros((num_nodes,)))
rho_array = rho_array.set(csdl.slice[0], rho_c) # cruise altitude
rho_array = rho_array.set(csdl.slice[1], rho_c) # cruise altitude

sos_array = csdl.Variable(value=np.zeros((num_nodes,)))
sos_array = sos_array.set(csdl.slice[0], a_c) # cruise altitude
sos_array = sos_array.set(csdl.slice[1], a_c) # cruise altitude

pitch_array = csdl.Variable(value=np.zeros(rho_array.shape))
pitch_array = pitch_array.set(csdl.slice[0], pitch)
pitch_array = pitch_array.set(csdl.slice[1], pitch[0] + dalpha_stab)

pitch_rad = pitch_array*np.pi/180.
# TO-DO: REMOVE AND INSERT V_INF AND PITCH INTO PANEL METHOD
V_rot_mat = csdl.Variable(value=0., shape=(num_nodes,3,3))
V_rot_mat = V_rot_mat.set(csdl.slice[:,1,1], value=1.)
V_rot_mat = V_rot_mat.set(csdl.slice[:,0,0], value=csdl.cos(pitch_rad))
V_rot_mat = V_rot_mat.set(csdl.slice[:,2,2], value=csdl.cos(pitch_rad))
V_rot_mat = V_rot_mat.set(csdl.slice[:,2,0], value=csdl.sin(pitch_rad))
V_rot_mat = V_rot_mat.set(csdl.slice[:,0,2], value=-csdl.sin(pitch_rad))

V_vec_rot = csdl.einsum(V_rot_mat, V_vec, action='ijk,ik->ij')
point_velocities = csdl.expand(V_vec_rot, (num_nodes,) + panel_mesh.shape[1:], 'ij->iaj')

# endregion

# region ============================ structures solver setup ============================
# structural mesh elements and nodes
num_half_nodes = num_nonwing_nodes + num_wing_nodes
num_half_el = num_half_nodes-1
num_beam_nodes = num_half_nodes*2 - 1
num_beam_el = num_beam_nodes-1

half_beam_mesh = (geometry.evaluate(LE_points_projected) + (geometry.evaluate(TE_points_projected)))/2.
width_fraction = 0.45
height_fraction = 0.65
projected_height = height_fraction*csdl.norm(
    geometry.evaluate(top_pts_projected) - geometry.evaluate(bot_pts_projected),
    axes=(1,)
)
projected_width = width_fraction*csdl.norm(
    geometry.evaluate(LE_points_projected)[1:,:] - geometry.evaluate(TE_points_projected)[1:,:],
    axes=(1,)
)

# beam mesh
beam_mesh = csdl.Variable(value=np.zeros((num_beam_nodes,3)))
beam_mesh = beam_mesh.set(csdl.slice[num_half_nodes-1:,:], value=half_beam_mesh)
beam_mesh = beam_mesh.set(csdl.slice[:num_half_nodes-1,0::2], value=half_beam_mesh[1:,0::2][::-1,:])
beam_mesh = beam_mesh.set(csdl.slice[:num_half_nodes-1,1], value=-half_beam_mesh[1:,1][::-1])

# beam thicknesses
half_ttop = np.ones((num_half_el,)) * 0.001
half_ttop = csdl.Variable(value=half_ttop)

ttop = csdl.Variable(value=np.zeros(num_beam_el)) # we assume symmetry on the top and bottom
ttop = ttop.set(csdl.slice[:num_half_el], value=half_ttop[::-1])
ttop = ttop.set(csdl.slice[num_half_el:], value=half_ttop)

half_tweb = np.ones((num_half_el,)) * 0.001
half_tweb = csdl.Variable(value=half_tweb)

tweb = csdl.Variable(value=np.zeros(num_beam_el))
tweb = tweb.set(csdl.slice[:num_half_el], value=half_tweb[::-1])
tweb = tweb.set(csdl.slice[num_half_el:], value=half_tweb)

# beam height
half_height = np.zeros((num_half_el,))
half_height = csdl.Variable(value=half_height)

height = csdl.Variable(value=np.zeros(num_beam_el))
height = height.set(csdl.slice[:num_half_el], value=projected_height[::-1])
height = height.set(csdl.slice[num_half_el:], value=projected_height)

height.add_name('beam_height')
height.save()

# beam width
half_width = np.zeros((num_half_el,))
half_width = csdl.Variable(value=half_width)

width = csdl.Variable(value=np.zeros(num_beam_el))
width = width.set(csdl.slice[:num_half_el], value=projected_width[::-1])
width = width.set(csdl.slice[num_half_el:], value=projected_width)

width.add_name('beam_width')
width.save()

# beam cross section
beam_CS = af.CSBox(ttop=ttop, tbot=ttop, tweb=tweb, height=height, width=width)

# beam material
aluminum = af.Material(name='aluminum', E=69E9, G=26E9, density=2700)

# beam OBJECT
BWB_beam = af.Beam(name='BWB_beam', mesh=beam_mesh, material=aluminum, cs=beam_CS)
BWB_beam_mass = BWB_beam.mass # property of beam
BWB_beam_mass.add_name('wing_mass')
BWB_beam_mass.save()
wing_weight_N = BWB_beam_mass * 9.81

cg_beam = BWB_beam.cg
cg_beam.add_name('wing_cg')
cg_beam.save()

mapper = NodalMap(normalization_eps=1.e-6) # force map from aero to beam
force_map = mapper.evaluate(panel_center, beam_mesh.reshape((-1,3)))
# endregion

# region ============================ mass properties ============================
# payload cg (same as centerbody centroid)
# LE_centerline = geometry.evaluate(LE_points_projected[0])
# TE_centerline = geometry.evaluate(TE_points_projected[0])
# payload_cg = (LE_centerline+TE_centerline)/2
payload_cg = oversized_payload_center_of_mass
payload_cg.add_name('payload_cg')
payload_cg.save()

# engine cg
engine_weight_lbs = 8760 # lbs
engine_weight_N = engine_weight_lbs*lbf_to_N
engine_diam = 2.67 # meters
engine_length = 4.24 # meters

engine_cg_evaluated = geometry.evaluate(engine_loc_parametric)
engine_sec_LE = geometry.evaluate(engine_sec_LE_parametric)
engine_sec_TE = geometry.evaluate(engine_sec_TE_parametric)

ecf = csdl.Variable(value=np.array([ecf_0])) # engine position chord fraction DV
engine_x_pos = engine_sec_TE[0]*ecf + engine_sec_LE[0]*(1-ecf)

engine_cg = csdl.Variable(value=engine_loc_0)
engine_cg = engine_cg.set(csdl.slice[0], engine_x_pos)
engine_cg = engine_cg.set(csdl.slice[1:], engine_cg_evaluated[1:])

engine_cg.add_name('engine_cg_neg')
engine_cg.save()

engine_cg_mirror = csdl.Variable(value=engine_loc_0)
engine_cg_mirror = engine_cg_mirror.set(csdl.slice[0], engine_x_pos)
engine_cg_mirror = engine_cg_mirror.set(csdl.slice[1], -engine_cg_evaluated[1])
engine_cg_mirror = engine_cg_mirror.set(csdl.slice[2], engine_cg_evaluated[2])

engine_cg_mirror.add_name('engine_cg_pos')
engine_cg_mirror.save()

# fuel cg (same as transition centroid)
transition_volume = 2*compute_volume(geometry, transition_volume_projection_points)
transition_volume.add_name('transition_volume')
transition_volume.save()

transition_LE = geometry.evaluate(transition_volume_LE_points_proj)
transition_TE = geometry.evaluate(transition_volume_TE_points_proj)

transition_avg = (transition_LE + transition_TE)/2
transition_cg = csdl.average(transition_avg, axes=(0,))

transition_cg.add_name('fuel_cg_pos')
transition_cg.save()

transition_cg_mirror = csdl.Variable(value=np.zeros(transition_cg.shape))
transition_cg_mirror = transition_cg_mirror.set(csdl.slice[:], transition_cg)
transition_cg_mirror = transition_cg_mirror.set(csdl.slice[1], -transition_cg[1])

transition_cg_mirror.add_name('fuel_cg_neg')
transition_cg_mirror.save()

fuel_cg = (transition_cg + transition_cg_mirror)/2. # dumb to do it this way but it's exact

# endregion

# region ============================ mission analysis ============================

'''
Computations for conditions:
- cruise conditions need drag approximations to measure L/D
- stability one only needs static margin
- structural conditions need none of these
'''
# panel_method

pm_solver_inputs = {
    'V_inf': point_velocities,
    'rho': rho_array,
    'sos': sos_array,
    'compressibility': True,
    'Cp cutoff': -5.,
    'partition_size': 1,
    'reuse_AIC': True,
    # 'mesh_path': file_path+file_name, # already done externally
    'ref_area': planform_area # does not matter bc we don't use the coefficients,
}
# we leave out the mesh path because we need FFD to move the mesh

panel_method = PanelMethod(
    solver_input_dict=pm_solver_inputs,
    skip_geometry=True # not running geometry
)
# inserting grid data from above
panel_method.insert_grid_data(
    mesh=panel_mesh[0,:],
    cell_adjacency_data=cell_adjacency_data,
    TE_properties=TE_properties
)

panel_method.declare_outputs([
    'Cp',
    'L',
    'Di',
    'M',
    'panel_forces'
])
outputs = panel_method.evaluate()


Cp = outputs['Cp']
Cp.add_name('Cp')
Cp.save()
L = outputs['L']
L.add_name('L')
L.save()
Di = outputs['Di']
Di.add_name('Di')
Di.save()
M = outputs['M'][:,1]
M.add_name('M_y')
M.save()
# panel_forces = outputs['panel_forces'][num_cruise:num_cruise+num_sizing,:]
panel_forces = outputs['panel_forces']#[0,:]

# computing coefficients
CL = L/(0.5*rho_array*V_inf_array**2*planform_area)
CL.add_name('CL')
CL.save()
CDi = Di/(0.5*rho_array*V_inf_array**2*planform_area)
CDi.add_name('CDi')
CDi.save()
CM = M/(0.5*rho_array*V_inf_array**2*planform_area*MAC)
CM.add_name('CMy')
CM.save()

# skin friction drag coefficient + strip theory
sec_chord_st = section_chords
sec_plan_area_st = section_spans*sec_chord_st
exp_shape = (1, sec_chord_st.shape[0])
sec_chord_st_nn = csdl.expand(sec_chord_st, exp_shape, 'i->ai')
sec_plan_area_st_nn = csdl.expand(sec_plan_area_st, exp_shape, 'i->ai')
V_cruise_array_exp = csdl.expand(V_cruise_array, exp_shape, 'i->ia')

Re=rho_c*V_cruise_array_exp*sec_chord_st_nn/mu_c
Cf = estimate_Cf(Re,M=0)
Cf.add_name('Cf')
Cf.save()
plan_2_wetted_area_conv = 2.1
wetted_area = sec_plan_area_st_nn*plan_2_wetted_area_conv # use 2x the sectional planform area here since we say flat plate
# FF = 1.5 # add a form factor equation here
FF = 1.5 # add a form factor equation here
Df_strip = 0.5 * rho_c*V_cruise_array_exp**2*wetted_area*Cf*FF
Df = csdl.sum(Df_strip, axes=(1,)) # multiply by two to account for other side
Df.add_name('Df')
Df.save()

# wave drag
t_c = (.15+.11336)/2
non_sectional_CDw = estimate_CDw(t_c, CL[:1], cruise_mach)
CDw, CDw_for_each_strip, mach_violation, Mcr, MDD, tech_component, thickness_component, lift_component = \
    estimate_sectional_CDw(section_t_c, CL[:1], cruise_mach, section_sweeps, planform_area_strips, planform_area)
CDw.add_name('CDw')
CDw.save()
Dw = 0.5*rho_array[:1]*V_cruise_array**2*planform_area*CDw
Dw.add_name('Dw')
Dw.save()

# fuel burn + takeoff weight calculations
L_cruise = L[:1]
D_cruise = Di[:1] + Df + Dw
L_D = L_cruise/D_cruise
L_D.add_name('L_D')
L_D.save()
TSFC = 0.355 # lb/lbf/hr
W2 = empty_weight_N_nowing+wing_weight_N+payload_weight_N
Wf, TOGW = estimate_fuel_burn(cruise_range_m, TSFC, V_cruise_array, L_D, W2)
# W_bar = empty_weight_N_nowing+wing_weight_N+payload_weight_N
# W_bar.add_name('W_no_fuel')
# W_bar.save()
# Wf, TOGW, W2 = estimate_fuel_burn_w_reserve(cruise_range_m, TSFC, V_cruise_array, L_D, W_bar)
Wf.add_name('Wf')
Wf.save()
TOGW.add_name('TOGW')
TOGW.save()
W2.add_name('W2')
W2.save()

fuel_burn = Wf

# structural sizing missions
beam_loads = csdl.Variable(value=np.zeros((num_beam_nodes, 6)))
# beam_loads = beam_loads.set(csdl.slice[:,:3], force_map.T() @ panel_forces[0,:])
beam_loads = beam_loads.set(csdl.slice[:,:3], force_map.T() @ panel_forces[0,:])

BWB_beam.fix(node=num_half_nodes-1)

BWB_beam.add_load(beam_loads)
frame_S1 = af.Frame(beams=[BWB_beam])
frame_S1.solve()
stress_dict_S1 = frame_S1.compute_stress()
beam_stress_S1 = stress_dict_S1[BWB_beam.name]
beam_stress_S1_MPa = beam_stress_S1/1.e6
beam_max_stress_S1_MPa = csdl.maximum(beam_stress_S1_MPa, rho=1e5) # NOTE: TUNE (seems to work)
beam_max_stress_S1 = beam_max_stress_S1_MPa * 1.e6
beam_max_stress_S1_SF = beam_max_stress_S1 * 1.5 * 2.5 # safety factor of 1.5, 2.5g pull-up




# cg computation (accounting for fuel weight)
TOGW_nn = csdl.Variable(shape=(num_nodes,), value=0.)
TOGW_nn = TOGW_nn.set(csdl.slice[:1], value=TOGW)
TOGW_nn = TOGW_nn.set(csdl.slice[1:1+0], value=TOGW[0])
# if do_stability:
TOGW_nn = TOGW_nn.set(csdl.slice[1], value=TOGW[0])

Wf_nn = csdl.Variable(shape=(num_nodes,), value=0.)
Wf_nn = Wf_nn.set(csdl.slice[:1], value=Wf)
Wf_nn = Wf_nn.set(csdl.slice[1:1+0], value=Wf[0])
# if do_stability:
Wf_nn = Wf_nn.set(csdl.slice[1], value=Wf[0])

payload_weight_nn = csdl.Variable(shape=(num_nodes,), value=0.)
payload_weight_nn = payload_weight_nn.set(csdl.slice[:1], value=payload_weight_N)
payload_weight_nn = payload_weight_nn.set(csdl.slice[1:1+0], value=payload_weight_N[0])
# if do_stability:
payload_weight_nn = payload_weight_nn.set(csdl.slice[1], value=payload_weight_N[0])

add_weight = TOGW_nn - wing_weight_N - 2*engine_weight_N - Wf_nn - payload_weight_nn # computing remaining weight at CG

target_shape = (num_nodes, 3)

cg_W_prod_wing = csdl.expand(wing_weight_N*cg_beam, target_shape, 'i->ai')
cg_W_prod_engine = csdl.expand(
    engine_weight_N*(engine_cg+engine_cg_mirror),
    target_shape,
    'i->ai'
)
cg_W_prod_fuel = csdl.expand(Wf_nn, target_shape, 'i->ia')/2. * csdl.expand(
    transition_cg+transition_cg_mirror,
    target_shape,
    'i->ai'
)
cg_W_prod_payload = csdl.expand(payload_weight_nn, target_shape, 'i->ia') * csdl.expand(
    payload_cg,
    target_shape,
    'i->ai'
)
cg_W_prod = cg_W_prod_wing+cg_W_prod_engine+cg_W_prod_fuel+cg_W_prod_payload

TOGW_exp = csdl.expand(TOGW_nn, target_shape, 'i->ia')
add_weight_exp = csdl.expand(add_weight, target_shape, 'i->ia')

cg = (cg_W_prod) / (TOGW_exp-add_weight_exp) # add_weight assumed to be at CG
cg.add_name('cg')
cg.save()
cg_x = cg[:,0]


cruise_trim = L_cruise - TOGW # force trim

# moment trim
aero_ref_pt = 0.
# CM_cg_cruise = CM[nominal_ind] + CL[nominal_ind]*(aero_ref_pt-cg_beam[0])/MAC
# CM_cg = CM + CL*(aero_ref_pt-cg_x)/MAC # WRONG
CM_cg = CM + CL*(cg_x-aero_ref_pt)/MAC
CM_cg.add_name('CM_cg')
CM_cg.save()

CM_cg_cruise_nominal = CM_cg[0]

# static margin
alpha_list = [pitch_array[0], pitch_array[1]] # pitch array is different from pitch dv
CL_list = [CL[0], CL[1]]
CM_list = [CM_cg[0], CM_cg[1]] # y-component taken above

static_margin = compute_static_margin(alpha_list, CL_list, CM_list)

neutral_point = static_margin*MAC + cg[0,0] # neutral point based on nominal condition
neutral_point.add_name('neutral_point')
neutral_point.save()
SM_missions = (neutral_point-cg[:,0])/MAC
SM_missions.add_name('static_margin_missions')
SM_missions.save()


# endregion

# region ============================ DVs ============================
@dataclass
class DVInfo:
    variable: csdl.Variable
    lower: float
    upper: float
    scaler: float = 1.0

"""
# all DV's for development
design_variables : dict[str, DVInfo]= {
    'pitch': DVInfo(variable=pitch, lower=-5, upper=5.),                                                                      # SP1
    'half_ttop': DVInfo(variable=half_ttop, lower=0.001, upper=0.1, scaler=1.e2),                                             # SP2
    'half_tweb': DVInfo(variable=half_tweb, lower=0.001, upper=0.1, scaler=1.e2),                                             # SP2
    'wing_twist_coefficients': DVInfo(variable=wing_twist_coefficients, lower=-10, upper=5),                                  # SP1
    'center_wing_half_span': DVInfo(variable=center_wing_half_span, lower=3, upper=8),                                        # SP1
    'transition_half_span': DVInfo(variable=transition_half_span, lower=3, upper=8),                                          # SP1
    'wing_half_span': DVInfo(variable=wing_half_span, lower=8, upper=22),                                                     # SP1
    'center_wing_chord_stretch_coefficients': DVInfo(variable=center_wing_chord_stretch_coefficients, lower=-center_wing_chords_computed.value*0.3, upper=center_wing_chords_computed.value*0.3), # SP1
    'wing_root_chord': DVInfo(variable=wing_root_chord, lower=0.5*wing_root_chord_computed.value, upper=2.*wing_root_chord_computed.value),                                                       # SP1
    'wing_tip_chord': DVInfo(variable=wing_tip_chord, lower=0.5*wing_tip_chord_computed.value, upper=2.*wing_tip_chord_computed.value),                                                           # SP1
    'wing_sweep': DVInfo(variable=wing_sweep, lower=0., upper=80., scaler=1.e-1),                                             # SP1
    'transition_sweep' : DVInfo(variable=transition_sweep, lower=-60., upper=80., scaler=1.e-1),                              # SP1
    'oversized_payload_translation_x': DVInfo(variable=oversized_payload_translation_x, lower=-10, upper=10),                 # SP2
    'oversized_payload_translation_z': DVInfo(variable=oversized_payload_translation_z, lower=-10, upper=10),                 # SP2
    'oversized_payload_rotation': DVInfo(variable=oversized_payload_rotation, lower=-6, upper=6),                           # SP2
    'cruise_trim_elevator_deflection': DVInfo(variable=cruise_trim_elevator_deflection, lower=-15., upper=45., scaler=1.e-1), # SP1
}
"""

# DV's for SP1
design_variables : dict[str, DVInfo]= {
    'pitch': DVInfo(variable=pitch, lower=-5, upper=5.),                                                                      # SP1
    # 'half_ttop': DVInfo(variable=half_ttop, lower=0.001, upper=0.1, scaler=1.e2),                                             # SP2
    # 'half_tweb': DVInfo(variable=half_tweb, lower=0.001, upper=0.1, scaler=1.e2),                                             # SP2
    'wing_twist_coefficients': DVInfo(variable=wing_twist_coefficients, lower=-10, upper=5),                                  # SP1
    'center_wing_half_span': DVInfo(variable=center_wing_half_span, lower=3, upper=8),                                        # SP1
    'transition_half_span': DVInfo(variable=transition_half_span, lower=3, upper=8),                                          # SP1
    'wing_half_span': DVInfo(variable=wing_half_span, lower=8, upper=22),                                                     # SP1
    'center_wing_chord_stretch_coefficients': DVInfo(variable=center_wing_chord_stretch_coefficients, lower=-center_wing_chords_computed.value*0.3, upper=center_wing_chords_computed.value*0.3), # SP1
    'wing_root_chord': DVInfo(variable=wing_root_chord, lower=0.5*wing_root_chord_computed.value, upper=2.*wing_root_chord_computed.value),                                                       # SP1
    'wing_tip_chord': DVInfo(variable=wing_tip_chord, lower=0.5*wing_tip_chord_computed.value, upper=2.*wing_tip_chord_computed.value),                                                           # SP1
    'wing_sweep': DVInfo(variable=wing_sweep, lower=0., upper=80., scaler=1.e-1),                                             # SP1
    'transition_sweep' : DVInfo(variable=transition_sweep, lower=-60., upper=80., scaler=1.e-1),                              # SP1
    # 'oversized_payload_translation_x': DVInfo(variable=oversized_payload_translation_x, lower=-10, upper=10),                 # SP2
    # 'oversized_payload_translation_z': DVInfo(variable=oversized_payload_translation_z, lower=-10, upper=10),                 # SP2
    # 'oversized_payload_rotation': DVInfo(variable=oversized_payload_rotation, lower=-6, upper=6),                           # SP2
    'cruise_trim_elevator_deflection': DVInfo(variable=cruise_trim_elevator_deflection, lower=-15., upper=45., scaler=1.e-1), # SP1
}
"""
# DV's for SP2
design_variables : dict[str, DVInfo]= {
    # 'pitch': DVInfo(variable=pitch, lower=-5, upper=5.),                                                                      # SP1
    'half_ttop': DVInfo(variable=half_ttop, lower=0.001, upper=0.1, scaler=1.e2),                                             # SP2
    'half_tweb': DVInfo(variable=half_tweb, lower=0.001, upper=0.1, scaler=1.e2),                                             # SP2
    # 'wing_twist_coefficients': DVInfo(variable=wing_twist_coefficients, lower=-10, upper=5),                                  # SP1
    # 'center_wing_half_span': DVInfo(variable=center_wing_half_span, lower=3, upper=8),                                        # SP1
    # 'transition_half_span': DVInfo(variable=transition_half_span, lower=3, upper=8),                                          # SP1
    # 'wing_half_span': DVInfo(variable=wing_half_span, lower=8, upper=22),                                                     # SP1
    # 'center_wing_chord_stretch_coefficients': DVInfo(variable=center_wing_chord_stretch_coefficients, lower=-center_wing_chords_computed.value*0.3, upper=center_wing_chords_computed.value*0.3), # SP1
    # 'wing_root_chord': DVInfo(variable=wing_root_chord, lower=0.5*wing_root_chord_computed.value, upper=2.*wing_root_chord_computed.value),                                                       # SP1
    # 'wing_tip_chord': DVInfo(variable=wing_tip_chord, lower=0.5*wing_tip_chord_computed.value, upper=2.*wing_tip_chord_computed.value),                                                           # SP1
    # 'wing_sweep': DVInfo(variable=wing_sweep, lower=0., upper=80., scaler=1.e-1),                                             # SP1
    # 'transition_sweep' : DVInfo(variable=transition_sweep, lower=-60., upper=80., scaler=1.e-1),                              # SP1
    'oversized_payload_translation_x': DVInfo(variable=oversized_payload_translation_x, lower=-10, upper=10),                 # SP2
    'oversized_payload_translation_z': DVInfo(variable=oversized_payload_translation_z, lower=-10, upper=10),                 # SP2
    'oversized_payload_rotation': DVInfo(variable=oversized_payload_rotation, lower=-6, upper=6),                           # SP2
    # 'cruise_trim_elevator_deflection': DVInfo(variable=cruise_trim_elevator_deflection, lower=-15., upper=45., scaler=1.e-1), # SP1
}
"""

for dv_name, dv_info in design_variables.items():
    dv_info.variable.set_as_design_variable(lower=dv_info.lower, upper=dv_info.upper, scaler=dv_info.scaler)
    dv_info.variable.add_name(dv_name)
# endregion

# # region ============================ constraints ============================

# # ==== trim constraints ====
# # cruise_trim.set_as_constraint(equals=0., scaler=1.e-6)
# cruise_trim.set_as_constraint(equals=0., scaler=1.e-5)
# cruise_trim.add_name('cruise_trim')

# CM_cg_cruise_nominal.set_as_constraint(equals=0., scaler=1.e1)
# CM_cg_cruise_nominal.add_name('cruise_cg_CM_nominal_trim_constraint')

# # static margin constraints
# # static_margin.set_as_constraint(lower=0.05, upper=0.5, scaler=1.e1)
# static_margin.set_as_constraint(lower=0.1, upper=0.5, scaler=1.e1)

# # max stress constraints
# beam_max_stress_S1_SF.set_as_constraint(upper=324e6, scaler=1.e-9)
# beam_max_stress_S1_SF.add_name('S1_max_stress')

# # geometric non-interference constraints
# oversized_payload_sdf_values.set_as_constraint(upper=0., scaler=1.)
# oversized_payload_sdf_values.add_name('oversized_payload_non_interference_constraint')
# # endregion

# # ============================ objective ============================
# fuel_burn.set_as_objective(scaler=1e-5)
# fuel_burn.add_name('fuel_burn_objective')


# ============================ augmented Lagrangian objective ============================
y = csdl.Variable(value=np.zeros(4))

c = csdl.Variable(value=np.zeros(4))
c = c.set(csdl.slice[0], cruise_trim * 1e-5)
c = c.set(csdl.slice[1], CM_cg_cruise_nominal * 1e1)
c = c.set(csdl.slice[2], (static_margin - 0.1) * 1e1)
c = c.set(csdl.slice[3], (beam_max_stress_S1_SF - 324e6) * 1e-9)

mu = csdl.Variable(value=50.)
augmented_lagrangian = 1e-5 * fuel_burn + csdl.inner(y, c) + 0.5 * mu * csdl.sum(c**2)
augmented_lagrangian.add_name('augmented_lagrangian')
augmented_lagrangian.set_as_objective()


# additional_inputs_dict = {'y': y, 'mu': mu}
additional_outputs_dict = {'c': c}
additional_outputs = list(additional_outputs_dict.values())
# additional_inputs = list(additional_inputs_dict.values())
# additional_outputs = list(additional_outputs_dict.values())

# additional inputs for SP1
additional_inputs_dict_SP1 = {
    'y': y,
    'mu': mu,
    'half_ttop': half_ttop,
    'half_tweb': half_tweb,
    'oversized_payload_translation_x': oversized_payload_translation_x,
    'oversized_payload_translation_z': oversized_payload_translation_z,
    'oversized_payload_rotation': oversized_payload_rotation,
}
additional_inputs_SP1 = list(additional_inputs_dict_SP1.values())

# additional inputs for SP2
additional_inputs_dict_SP2 = {
    'y': y,
    'mu': mu,
    'pitch': pitch,
    'wing_twist_coefficients': wing_twist_coefficients,
    'center_wing_half_span': center_wing_half_span,
    'transition_half_span': transition_half_span,
    'wing_half_span': wing_half_span,
    'center_wing_chord_stretch_coefficients': center_wing_chord_stretch_coefficients,
    'wing_root_chord': wing_root_chord,
    'wing_tip_chord': wing_tip_chord,
    'wing_sweep': wing_sweep,
    'transition_sweep': transition_sweep,
    'cruise_trim_elevator_deflection': cruise_trim_elevator_deflection
}
additional_inputs_SP2 = list(additional_inputs_dict_SP2.values())


# additional_outputs : list[csdl.Variable] = [c]
# additional_outputs : list[csdl.Variable] = []
# additional_outputs += [func.coefficients for func in geometry.functions.values()]
# additional_outputs += [func.coefficients for func in oversized_payload_geometry.functions.values()]
# additional_outputs += [fuel_burn, static_margin, CM_cg_cruise_nominal, cruise_trim, TOGW, Wf, L_D, CL, CDw, L, Di, Df, Dw, L_cruise, D_cruise, non_sectional_CDw, section_spans, section_chords, section_sweeps, section_t_c]
# additional_outputs += [CDw_for_each_strip, mach_violation, Mcr, MDD, tech_component, thickness_component, lift_component]

print('Model 1 checkpoint')
fname = f'aero_structural_opt_SLSQP_1_missions'
sim_1 = csdl.experimental.JaxSimulator(recorder,
                                     additional_inputs=additional_inputs_SP1, # !!!!! CHANGE FOR SP1 OR SP2 !!!!!
                                     additional_outputs=additional_outputs,
                                    #  save_on_update=True, 
                                     filename=fname, 
                                     output_saved=True)