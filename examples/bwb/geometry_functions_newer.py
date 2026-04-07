import csdl_alpha as csdl
import lsdo_geo
import lsdo_function_spaces as lfs
import numpy as np
import time

def initial_refit(geometry: lsdo_geo.Geometry, plot:bool=False):
    wing_c_indices = [0,1,8,9]  # lower right, upper right, lower left, upper left
    # wing_r_transition_indices = [2,3]   # lower, upper
    # wing_r_indices = [4,5,6,7]  # lower, upper, lower tip, upper tip
    wing_l_transition_indices = [10,11] # lower, upper
    wing_l_indices = [12,13,14,15] # lower, upper, lower tip, upper tip

    left_wing_transition = geometry.declare_component(wing_l_transition_indices)
    left_wing = geometry.declare_component(wing_l_indices)
    # right_wing_transition = geometry.declare_component(wing_r_transition_indices)
    # right_wing = geometry.declare_component(wing_r_indices)
    center_wing = geometry.declare_component(wing_c_indices)
    # oml = geometry.declare_component(oml_indices)

    # keep chordwise resolution the same as it initially is for the wing for now
    # chordwise_resolution = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]//5
    chordwise_resolution = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]//2
    # Use what we were refitting before for chordwise resolution for wing, keep others original for now
    spanwise_resolution_wing = 9
    # spanwise_resolution_wing = int(30/2)  # preferred option so far
    # spanwise_resolution_wing = int(30)
    # spanwise_resolution_wing = int(30*2)
    spanwise_resolution_transition = int(left_wing_transition.functions[wing_l_transition_indices[0]].coefficients.shape[0]/2)
    # spanwise_resolution_transition = int(left_wing_transition.functions[wing_l_transition_indices[0]].coefficients.shape[0])
    # spanwise_resolution_transition = int(left_wing_transition.functions[wing_l_transition_indices[0]].coefficients.shape[0]*2)
    spanwise_resolution_center_wing = int(center_wing.functions[wing_c_indices[0]].coefficients.shape[0]/2)
    # spanwise_resolution_center_wing = int(center_wing.functions[wing_c_indices[0]].coefficients.shape[0])
    spanwise_resolutions = [spanwise_resolution_wing, spanwise_resolution_transition, spanwise_resolution_center_wing, 
                            spanwise_resolution_center_wing, spanwise_resolution_transition, spanwise_resolution_wing]
    num_coefficients_spanwise = spanwise_resolution_wing*2+spanwise_resolution_transition*2+spanwise_resolution_center_wing*2 + 1 # extra one for the center so there are coefficients in the middle
    oml_surface_b_spline_space = lfs.BSplineSpaceNew(
        num_parametric_dimensions=2,
        degree=(3,3),
        coefficients_shape=(chordwise_resolution, num_coefficients_spanwise)
        # coefficients_shape=(num_coefficients_spanwise, chordwise_resolution)
    )
    # full_oml_b_spline_space = lfs.BSplineSpace(
    #     num_parametric_dimensions=2,
    #     degree=(3,3),
    #     coefficients_shape=(2*chordwise_resolution, num_coefficients_spanwise)
    #     # coefficients_shape=(num_coefficients_spanwise, 2*chordwise_resolution)
    # )
    lower_indices = [12, 10, 8, 0, 2, 4] # left wing, left wing transition, left center wing, right center wing, right wing transition, right wing
    upper_indices = [13, 11, 9, 1, 3, 5] # left wing, left wing transition, left center wing, right center wing, right wing transition, right wing

    # Left-side patches need opposite span direction compared to right-side patches
    # for a globally consistent left->right parameter direction when stitching.
    left_lower_indices = set(lower_indices[:3])
    left_upper_indices = set(upper_indices[:3])

    # discretize each surface
    # grid_resolution = (300,35) # (chordwise, spanwise)
    # grid_resolution = (200,17) # (chordwise, spanwise)
    grid_resolution = (200,11) # (chordwise, spanwise)
    # grid_resolution = (300,65) # (chordwise, spanwise)

    lower_discretizations = []
    start_value = 0
    for i, idx in enumerate(lower_indices):
        function = geometry.functions[idx]
        # Currently, all functions have u as spanwise and v as chordwise
        # - For both, u=0 is root side. For left, v=0 is leading edge, for right, v=0 is trailing edge
        parametric_grid = function.space.generate_parametric_grid(grid_resolution=(grid_resolution[0], grid_resolution[1])).copy()
        # Original convention for all functions:
        #   - param 0: spanwise (root->tip)
        #   - param 1: chordwise
        # Additional note:
        #   - left surfaces: param1=0 is LE
        #   - right surfaces: param1=0 is TE
        #
        # Transform so evaluations are aligned across all stitched patches:
        #   - aligned u: chordwise LE->TE
        #   - aligned v: spanwise left->right
        parametric_grid_flipped_uv = parametric_grid.copy()
        parametric_grid_flipped_uv[:, 0] = parametric_grid[:, 1]
        parametric_grid_flipped_uv[:, 1] = parametric_grid[:, 0]
        parametric_grid = parametric_grid_flipped_uv
        
        u_orig = parametric_grid[:, 0].copy()  # spanwise
        v_orig = parametric_grid[:, 1].copy()  # chordwise

        if idx in left_lower_indices:
            # Left side: reverse span so it increases left->right in global space.
            parametric_grid[:, 0] = 1.0 - u_orig
            # Left side already has LE->TE for chordwise parameter.
            parametric_grid[:, 1] = v_orig
        else:
            # Right side: keep span as-is (root->tip is left->right globally).
            parametric_grid[:, 0] = u_orig
            # Right side chordwise is TE->LE, so flip to LE->TE.
            parametric_grid[:, 1] = 1.0 - v_orig
        
        discretization = function.evaluate(parametric_grid).value.reshape((grid_resolution[0], grid_resolution[1], 3))

        mesh_grid_input = []
        mesh_grid_input.append(np.linspace(0., 1., grid_resolution[0]))
        if i == 2 or i == 3: # center wing, add an extra spanwise section for the middle
            # end_value = 0.5
            end_value = start_value + (spanwise_resolutions[i]+0.5)/num_coefficients_spanwise
        else:
            end_value = start_value + spanwise_resolutions[i]/num_coefficients_spanwise
        mesh_grid_input.append(np.linspace(start_value, end_value, grid_resolution[1]))
        start_value = end_value
        parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
        for dimensions_index in range(2):
            parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))
        parametric_coordinates = np.hstack(parametric_coordinates_tuple)

        lower_discretizations.append((parametric_coordinates, discretization))

    upper_discretizations = []
    start_value = 0
    spanwise_section_boundaries = [0]
    for i, idx in enumerate(upper_indices):
        function = geometry.functions[idx]
        # Currently, all functions have u as spanwise and v as chordwise
        # - For both, u=0 is root side. For left, v=0 is trailing edge, for right, v=0 is leading edge
        parametric_grid = function.space.generate_parametric_grid(grid_resolution=(grid_resolution[0], grid_resolution[1])).copy()
        # Original convention for all functions:
        #   - param 0: spanwise (root->tip)
        #   - param 1: chordwise
        # Additional note:
        #   - left surfaces: param1=0 is TE
        #   - right surfaces: param1=0 is LE
        #
        # Transform so evaluations are aligned across all stitched patches:
        #   - aligned u: chordwise LE->TE
        #   - aligned v: spanwise left->right
        parametric_grid_flipped_uv = parametric_grid.copy()
        parametric_grid_flipped_uv[:, 0] = parametric_grid[:, 1]
        parametric_grid_flipped_uv[:, 1] = parametric_grid[:, 0]
        parametric_grid = parametric_grid_flipped_uv

        u_orig = parametric_grid[:, 0].copy()  # spanwise
        v_orig = parametric_grid[:, 1].copy()  # chordwise

        if idx in left_upper_indices:
            # Left side: reverse span so it increases left->right in global space.
            parametric_grid[:, 0] = 1.0 - u_orig
            # Left side chordwise is TE->LE, so flip to LE->TE.
            parametric_grid[:, 1] = 1.0 - v_orig
        else:
            # Right side: keep span as-is (root->tip is left->right globally).
            parametric_grid[:, 0] = u_orig
            # Right side already has LE->TE for chordwise parameter.
            parametric_grid[:, 1] = v_orig

        discretization = function.evaluate(parametric_grid).value.reshape((grid_resolution[0], grid_resolution[1], 3))

        mesh_grid_input = []
        mesh_grid_input.append(np.linspace(0., 1., grid_resolution[0]))
        if i == 2 or i == 3: # center wing, add an extra spanwise section for the middle
            # end_value = 0.5
            end_value = start_value + (spanwise_resolutions[i]+0.5)/num_coefficients_spanwise
        else:
            end_value = start_value + spanwise_resolutions[i]/num_coefficients_spanwise
        spanwise_section_boundaries.append(end_value)
        mesh_grid_input.append(np.linspace(start_value, end_value, grid_resolution[1]))
        start_value = end_value
        parametric_coordinates_tuple = np.meshgrid(*mesh_grid_input, indexing='ij')
        for dimensions_index in range(2):
            parametric_coordinates_tuple[dimensions_index] = parametric_coordinates_tuple[dimensions_index].reshape((-1,1))
        parametric_coordinates = np.hstack(parametric_coordinates_tuple)

        upper_discretizations.append((parametric_coordinates, discretization))

    # stitch them together and refit to the new B-spline space
    lower_parametric_coordinates = []
    lower_points = []
    for discretization in lower_discretizations:
        lower_parametric_coordinates.append(discretization[0])
        lower_points.append(discretization[1].reshape((-1, 3)))

    lower_parametric_coordinates = np.vstack(lower_parametric_coordinates)
    # lower_points = np.vstack(lower_points)
    lower_points = csdl.vstack(lower_points)

    upper_parametric_coordinates = []
    upper_points = []
    for discretization in upper_discretizations:
        upper_parametric_coordinates.append(discretization[0])
        upper_points.append(discretization[1].reshape((-1, 3)))

    upper_parametric_coordinates = np.vstack(upper_parametric_coordinates)
    # upper_points = np.vstack(upper_points)
    upper_points = csdl.vstack(upper_points)

    # For repeated seam coordinates, keep one coordinate and average its point values.
    def deduplicate_and_average_points(parametric_coordinates, points):
        unique_coords, inverse = np.unique(parametric_coordinates, return_inverse=True, axis=0)
        counts = np.bincount(inverse)
        accumulated_points = np.zeros((unique_coords.shape[0], 3))
        np.add.at(accumulated_points, inverse, points)
        averaged_points = accumulated_points / counts[:, None]
        return unique_coords, averaged_points

    # lower_parametric_coordinates, lower_points = deduplicate_and_average_points(
    #     lower_parametric_coordinates,
    #     lower_points,
    # )
    # upper_parametric_coordinates, upper_points = deduplicate_and_average_points(
    #     upper_parametric_coordinates,
    #     upper_points,
    # )

    # upper_parametric_coordinates_for_full_oml = upper_parametric_coordinates.copy()
    # lower_parametric_coordinates_for_full_oml = lower_parametric_coordinates.copy()
    # upper_parametric_coordinates_for_full_oml[:,0] = (1.0 - upper_parametric_coordinates[:,0])/2
    # lower_parametric_coordinates_for_full_oml[:,0] = (lower_parametric_coordinates[:,0]/2) + 0.5

    # new_oml = full_oml_b_spline_space.fit_function(
    #     values=np.concatenate((lower_points, upper_points), axis=0), 
    #     parametric_coordinates=np.concatenate((lower_parametric_coordinates_for_full_oml, upper_parametric_coordinates_for_full_oml), axis=0), 
    #     regularization_parameter=None
    # )

    basis_matrix = oml_surface_b_spline_space.compute_basis_matrix(lower_parametric_coordinates)
    fitting_matrix = basis_matrix.T @ basis_matrix
    fitting_matrix = fitting_matrix.tocsc()
    sparse_inverse_times = []
    lower_inverted_fitting_matrix = np.linalg.inv(fitting_matrix.toarray())

    # rhs = basis_matrix.T @ csdl.Variable(value=lower_points)
    rhs = basis_matrix.T @ lower_points
    lower_oml_coefficients = (lower_inverted_fitting_matrix @ rhs).reshape((chordwise_resolution, num_coefficients_spanwise, 3))

    basis_matrix = oml_surface_b_spline_space.compute_basis_matrix(upper_parametric_coordinates)
    fitting_matrix = basis_matrix.T @ basis_matrix
    fitting_matrix = fitting_matrix.tocsc()
    # inverted_fitting_matrix = spla.inv(fitting_matrix).toarray()
    upper_inverted_fitting_matrix = np.linalg.inv(fitting_matrix.toarray())
    # rhs = basis_matrix.T @ csdl.Variable(value=upper_points)
    rhs = basis_matrix.T @ upper_points
    upper_oml_coefficients = (upper_inverted_fitting_matrix @ rhs).reshape((chordwise_resolution, num_coefficients_spanwise, 3))

    lower_oml = lfs.Function(space=oml_surface_b_spline_space, coefficients=lower_oml_coefficients)
    upper_oml = lfs.Function(space=oml_surface_b_spline_space, coefficients=upper_oml_coefficients)

    # Create wing caps using coefficients at each tip
    wing_cap_b_spline_space = lfs.BSplineSpaceNew(
        num_parametric_dimensions=2,
        degree=(3,1),
        coefficients_shape=(chordwise_resolution, 2)
    )
    left_wing_cap_coefficients_lower = lower_oml_coefficients[:, 0]
    left_wing_cap_coefficients_upper = upper_oml_coefficients[:, 0]
    left_wing_cap_coefficients = csdl.Variable(shape=(chordwise_resolution, 2, 3), value=0.)
    left_wing_cap_coefficients = left_wing_cap_coefficients.set(csdl.slice[:, 0, :], left_wing_cap_coefficients_lower)
    left_wing_cap_coefficients = left_wing_cap_coefficients.set(csdl.slice[:, 1, :], left_wing_cap_coefficients_upper)
    left_wing_cap = lfs.Function(space=wing_cap_b_spline_space, coefficients=left_wing_cap_coefficients)
    right_wing_cap_coefficients_lower = lower_oml_coefficients[:, -1]
    right_wing_cap_coefficients_upper = upper_oml_coefficients[:, -1]
    right_wing_cap_coefficients = csdl.Variable(shape=(chordwise_resolution, 2, 3), value=0.)
    right_wing_cap_coefficients = right_wing_cap_coefficients.set(csdl.slice[:, 0, :], right_wing_cap_coefficients_lower)
    right_wing_cap_coefficients = right_wing_cap_coefficients.set(csdl.slice[:, 1, :], right_wing_cap_coefficients_upper)
    right_wing_cap = lfs.Function(space=wing_cap_b_spline_space, coefficients=right_wing_cap_coefficients)

    new_geometry = lsdo_geo.Geometry(functions={0: lower_oml, 1: upper_oml, 2: left_wing_cap, 3: right_wing_cap})


    if plot:
        geometry_plot = geometry.plot(show=False, opacity=0.5)
        new_geometry.plot(additional_plotting_elements=geometry_plot, color='red', opacity=0.8)

    return new_geometry, lower_discretizations, upper_discretizations, \
        lower_parametric_coordinates, upper_parametric_coordinates, \
        lower_inverted_fitting_matrix, upper_inverted_fitting_matrix, \
        spanwise_section_boundaries


def setup_and_evaluate_geometry_parameterization(geometry: lsdo_geo.Geometry, variables: dict[str, csdl.Variable],
    lower_discretizations, upper_discretizations, 
    lower_parametric_coordinates, upper_parametric_coordinates,
    lower_inverted_fitting_matrix, upper_inverted_fitting_matrix, 
    spanwise_section_boundaries, plot:bool=False) -> lsdo_geo.Geometry:

    enforce_interface_smoothing = False
    enforce_no_centerbody_trailing_edge_sweep = True
    enforce_thickness_to_chord_ratio = True

    wing_c_indices = [0,1,8,9]  # lower right, upper right, lower left, upper left
    # wing_r_transition_indices = [2,3]   # lower, upper
    # wing_r_indices = [4,5,6,7]  # lower, upper, lower tip, upper tip
    wing_l_transition_indices = [10,11] # lower, upper
    wing_l_indices = [12,13,14,15] # lower, upper, lower tip, upper tip

    left_wing_transition = geometry.declare_component(wing_l_transition_indices)
    left_wing = geometry.declare_component(wing_l_indices)
    # right_wing_transition = geometry.declare_component(wing_r_transition_indices)
    # right_wing = geometry.declare_component(wing_r_indices)
    center_wing = geometry.declare_component(wing_c_indices)

    # keep chordwise resolution the same as it initially is for the wing for now
    # chordwise_resolution = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]//5
    chordwise_resolution = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]//2
    # Use what we were refitting before for chordwise resolution for wing, keep others original for now
    spanwise_resolution_wing = 9
    # spanwise_resolution_wing = int(30/2)  # no folds
    # spanwise_resolution_wing = int(30)
    # spanwise_resolution_wing = int(30*2)
    spanwise_resolution_transition = int(left_wing_transition.functions[wing_l_transition_indices[0]].coefficients.shape[0]/2)
    # spanwise_resolution_transition = int(left_wing_transition.functions[wing_l_transition_indices[0]].coefficients.shape[0])
    # spanwise_resolution_transition = int(left_wing_transition.functions[wing_l_transition_indices[0]].coefficients.shape[0]*2)
    spanwise_resolution_center_wing = int(center_wing.functions[wing_c_indices[0]].coefficients.shape[0]/2)
    # spanwise_resolution_center_wing = int(center_wing.functions[wing_c_indices[0]].coefficients.shape[0])
    spanwise_resolutions = [spanwise_resolution_wing, spanwise_resolution_transition, spanwise_resolution_center_wing, 
                            spanwise_resolution_center_wing, spanwise_resolution_transition, spanwise_resolution_wing]
    num_coefficients_spanwise = spanwise_resolution_wing*2+spanwise_resolution_transition*2+spanwise_resolution_center_wing*2 + 1 # extra one for the center so there are coefficients in the middle
    oml_surface_b_spline_space = lfs.BSplineSpaceNew(
        num_parametric_dimensions=2,
        degree=(3,3),
        coefficients_shape=(chordwise_resolution, num_coefficients_spanwise)
        # coefficients_shape=(num_coefficients_spanwise, chordwise_resolution)
    )

    # discretize each surface
    # grid_resolution = (300,35) # (chordwise, spanwise)
    # grid_resolution = (200,17) # (chordwise, spanwise)
    grid_resolution = (200,11) # (chordwise, spanwise)
    # grid_resolution = (300,65) # (chordwise, spanwise)


    num_center_wing_spanwise_design_dofs = 4
    # num_wing_twist_design_variables = 4
    num_wing_twist_design_variables = 2
    parameterization_solver = lsdo_geo.ParameterizationSolver()

    right_wing_volume_mesh = csdl.Variable(shape=lower_discretizations[-1][1].shape[:2]+(2,3), value=0.)
    right_wing_volume_mesh = right_wing_volume_mesh.set(csdl.slice[:,:,0,:], lower_discretizations[-1][1])
    right_wing_volume_mesh = right_wing_volume_mesh.set(csdl.slice[:,:,1,:], upper_discretizations[-1][1])
    right_transition_volume_mesh = csdl.Variable(shape=lower_discretizations[-2][1].shape[:2]+(2,3), value=0.)
    right_transition_volume_mesh = right_transition_volume_mesh.set(csdl.slice[:,:,0,:], lower_discretizations[-2][1])
    right_transition_volume_mesh = right_transition_volume_mesh.set(csdl.slice[:,:,1,:], upper_discretizations[-2][1])
    right_center_wing_volume_mesh = csdl.Variable(shape=lower_discretizations[-3][1].shape[:2]+(2,3), value=0.)
    right_center_wing_volume_mesh = right_center_wing_volume_mesh.set(csdl.slice[:,:,0,:], lower_discretizations[-3][1])
    right_center_wing_volume_mesh = right_center_wing_volume_mesh.set(csdl.slice[:,:,1,:], upper_discretizations[-3][1])

    right_wing_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(right_wing_volume_mesh, principal_parametric_dimension=1)
    right_transition_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(right_transition_volume_mesh, principal_parametric_dimension=1)
    right_center_wing_sectional_parameterization = lsdo_geo.VolumeSectionalParameterization(right_center_wing_volume_mesh, principal_parametric_dimension=1)

    wing_section_centers = []
    for i in range(right_wing_volume_mesh.shape[1]):
        section_coefficients = right_wing_volume_mesh[:,i,:,:].value
        section_control_points = section_coefficients.reshape(-1, 3)
        section_center = np.mean(section_control_points, axis=0)
        wing_section_centers.append(section_center)
    center_wing_section_centers = []
    for i in range(right_center_wing_volume_mesh.shape[1]):
        section_coefficients = right_center_wing_volume_mesh[:,i,:,:].value
        section_control_points = section_coefficients.reshape(-1, 3)
        section_center = np.mean(section_control_points, axis=0)
        center_wing_section_centers.append(section_center)
    transition_section_centers = []
    for i in range(right_transition_volume_mesh.shape[1]):
        section_coefficients = right_transition_volume_mesh[:,i,:,:].value
        section_control_points = section_coefficients.reshape(-1, 3)
        section_center = np.mean(section_control_points, axis=0)
        transition_section_centers.append(section_center)

    wing_half_span = wing_section_centers[-1][1] - wing_section_centers[0][1]
    wing_linear_interpolation_coefficients = 1 - (np.array(wing_section_centers)[:,1] - (wing_section_centers[0][1])) / wing_half_span

    center_wing_half_span = center_wing_section_centers[-1][1] - center_wing_section_centers[0][1]
    center_wing_linear_interpolation_coefficients = 1 - (np.array(center_wing_section_centers)[:,1] - (center_wing_section_centers[0][1])) / center_wing_half_span

    transition_half_span = transition_section_centers[-1][1] - transition_section_centers[0][1]
    transition_linear_interpolation_coefficients = 1 - (np.array(transition_section_centers)[:,1] - (transition_section_centers[0][1])) / transition_half_span


    # center_wing_parameters_b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(center_wing_spanwise_design_dofs,))
    center_wing_parameters_b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(num_center_wing_spanwise_design_dofs*2-1,))
    transition_num_spanwise_states = 4
    transition_parameters_b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(transition_num_spanwise_states,))
    # wing_twist_b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=3, coefficients_shape=(num_wing_twist_design_variables,))
    wing_twist_b_spline_space = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(num_wing_twist_design_variables,))


    wing_tip_sweep_translation = csdl.Variable(value=0.)
    # center_wing_tip_sweep_translation = csdl.Variable(value=0.)
    center_wing_sweep_translation_dofs = csdl.Variable(shape=(num_center_wing_spanwise_design_dofs,), value=0.)
    center_wing_sweep_translation_coefficients = csdl.concatenate((center_wing_sweep_translation_dofs[::-1], center_wing_sweep_translation_dofs[1:]), axis=0)
    transition_tip_sweep_translation = csdl.Variable(value=0.)
    transition_root_sweep_translation = center_wing_sweep_translation_coefficients[-1]
    transition_sweep_translation_coefficients = csdl.concatenate((transition_root_sweep_translation.reshape(1,), transition_root_sweep_translation.reshape(1,), transition_tip_sweep_translation.reshape(1,), transition_tip_sweep_translation.reshape(1,)), axis=0)
    wing_root_sweep_translation = transition_tip_sweep_translation

    wing_sweep_translations = csdl.linear_combination(wing_root_sweep_translation, wing_tip_sweep_translation, num_steps=grid_resolution[1],
                                                    start_weights=wing_linear_interpolation_coefficients, stop_weights=1-wing_linear_interpolation_coefficients).flatten()
    # center_wing_sweep_translations = csdl.linear_combination(0., center_wing_tip_sweep_translation, num_steps=grid_resolution[1],
    #                                                     start_weights=center_wing_linear_interpolation_coefficients, stop_weights=1-center_wing_linear_interpolation_coefficients).flatten()
    center_wing_sweep_translations_b_spline = lfs.Function(space=center_wing_parameters_b_spline_space, coefficients=center_wing_sweep_translation_coefficients.reshape((num_center_wing_spanwise_design_dofs*2-1,1)))
    center_wing_sweep_translations = center_wing_sweep_translations_b_spline.evaluate((1-center_wing_linear_interpolation_coefficients)/2+0.5, plot=False).flatten()

    
    # transition_sweep_translations = csdl.linear_combination(transition_root_sweep_translation, transition_tip_sweep_translation, num_steps=grid_resolution[1],
    #                                                     start_weights=transition_linear_interpolation_coefficients, stop_weights=1-transition_linear_interpolation_coefficients).flatten()
    # transition_sweep_translations = center_wing_chord_stretch_dofs[-1]
    # transition_stretch_coefficients = csdl.Variable(shape=(transition_num_spanwise_states,), value=0.)
    # transition_sweep_translation_internal_coefficients = csdl.Variable(shape=(transition_num_spanwise_states-2,), value=0.)
    # transition_sweep_translation_coefficients = csdl.concatenate((transition_root_sweep_translation.reshape(1,), transition_sweep_translation_internal_coefficients, transition_tip_sweep_translation.reshape(1,)), axis=0)
    transition_sweep_b_spline = lfs.Function(space=transition_parameters_b_spline_space, coefficients=transition_sweep_translation_coefficients)
    transition_sweep_translations = transition_sweep_b_spline.evaluate(1-transition_linear_interpolation_coefficients, plot=False).flatten()


    wing_tip_wingspan_stretch = csdl.Variable(value=0.)
    center_wing_tip_wingspan_stretch = csdl.Variable(value=0.)
    transition_tip_wingspan_stretch = csdl.Variable(value=0.)
    transition_root_wingspan_stretch = center_wing_tip_wingspan_stretch
    wing_root_wingspan_stretch = transition_tip_wingspan_stretch

    wingspan_stretch_axis = csdl.Variable(value=np.array([0., 1., 0.]))
    # wingspan_stretch_axis = np.array([0., 1., 0.])
    # twist_axis = csdl.Variable(value=np.array([0., 1., 0.]))
    twist_axis = np.array([0., 1., 0.])
    wing_wingspan_stretches = csdl.linear_combination(wing_root_wingspan_stretch, wing_tip_wingspan_stretch, num_steps=grid_resolution[1],
                                                    start_weights=wing_linear_interpolation_coefficients, stop_weights=1-wing_linear_interpolation_coefficients).flatten()
    center_wing_wingspan_stretches = csdl.linear_combination(0., center_wing_tip_wingspan_stretch, num_steps=grid_resolution[1],
                                                        start_weights=center_wing_linear_interpolation_coefficients, stop_weights=1-center_wing_linear_interpolation_coefficients).flatten()
    transition_wingspan_stretches = csdl.linear_combination(transition_root_wingspan_stretch, transition_tip_wingspan_stretch, num_steps=grid_resolution[1],
                                                        start_weights=transition_linear_interpolation_coefficients, stop_weights=1-transition_linear_interpolation_coefficients).flatten()

    wing_tip_dihedral_translation = csdl.Variable(value=0.)
    center_wing_tip_dihedral_translation = csdl.Variable(value=0.)
    transition_tip_dihedral_translation = csdl.Variable(value=0.)
    transition_root_dihedral_translation = center_wing_tip_dihedral_translation
    wing_root_dihedral_translation = transition_tip_dihedral_translation

    wing_dihedral_translations = csdl.linear_combination(wing_root_dihedral_translation, wing_tip_dihedral_translation, num_steps=grid_resolution[1],
                                                    start_weights=wing_linear_interpolation_coefficients, stop_weights=1-wing_linear_interpolation_coefficients).flatten()
    center_wing_dihedral_translations = csdl.linear_combination(0., center_wing_tip_dihedral_translation, num_steps=grid_resolution[1],
                                                        start_weights=center_wing_linear_interpolation_coefficients, stop_weights=1-center_wing_linear_interpolation_coefficients).flatten()
    transition_dihedral_translations = csdl.linear_combination(transition_root_dihedral_translation, transition_tip_dihedral_translation, num_steps=grid_resolution[1],
                                                        start_weights=transition_linear_interpolation_coefficients, stop_weights=1-transition_linear_interpolation_coefficients).flatten()

    wing_root_chord_stretch = csdl.Variable(value=0.)
    wing_tip_chord_stretch = csdl.Variable(value=0.)
    # center_wing_root_chord_stretch = csdl.Variable(value=0.)
    # center_wing_tip_chord_stretch = csdl.Variable(value=0.)
    if 'center_wing_chord_stretch_coefficients' in variables:
        center_wing_chord_stretch_dofs = variables['center_wing_chord_stretch_coefficients']
    else:
        center_wing_chord_stretch_dofs = csdl.Variable(shape=(num_center_wing_spanwise_design_dofs,), value=0.)
    # center_wing_chord_stretch_dofs = csdl.Variable(shape=(center_wing_spanwise_design_dofs,), value=np.array([0., 0., 0., -1.]))
    center_wing_chord_stretch_coefficients = csdl.concatenate((center_wing_chord_stretch_dofs[::-1], center_wing_chord_stretch_dofs[1:]), axis=0)
    # transition_root_chord_stretch = center_wing_tip_chord_stretch
    transition_root_chord_stretch = center_wing_chord_stretch_dofs[-1]
    transition_tip_chord_stretch = wing_root_chord_stretch
    # transition_stretch_coefficients = csdl.Variable(shape=(transition_num_spanwise_states,), value=0.)
    # transition_stretch_internal_coefficients = csdl.Variable(shape=(transition_num_spanwise_states-2,), value=0.)
    # transition_stretch_coefficients = csdl.concatenate((transition_root_chord_stretch.reshape(1,), transition_stretch_internal_coefficients, transition_tip_chord_stretch.reshape(1,)), axis=0)
    transition_chord_stretch_coefficients = csdl.concatenate((transition_root_chord_stretch.reshape(1,), transition_root_chord_stretch.reshape(1,), transition_tip_chord_stretch.reshape(1,), transition_tip_chord_stretch.reshape(1,)), axis=0)

    wing_chord_stretches = csdl.linear_combination(wing_root_chord_stretch, wing_tip_chord_stretch, num_steps=grid_resolution[1],
                                                start_weights=wing_linear_interpolation_coefficients, stop_weights=1-wing_linear_interpolation_coefficients).flatten()
    # center_wing_chord_stretches = csdl.linear_combination(center_wing_root_chord_stretch, center_wing_tip_chord_stretch, num_steps=grid_resolution[1],
    #                                                start_weights=center_wing_linear_interpolation_coefficients, stop_weights=1-center_wing_linear_interpolation_coefficients).flatten()
    center_wing_chord_stretches_b_spline = lfs.Function(space=center_wing_parameters_b_spline_space, coefficients=center_wing_chord_stretch_coefficients.reshape((num_center_wing_spanwise_design_dofs*2-1,1)))
    # center_wing_chord_stretches = center_wing_chord_stretches_b_spline.evaluate(1-center_wing_linear_interpolation_coefficients, plot=False).flatten()
    center_wing_chord_stretches = center_wing_chord_stretches_b_spline.evaluate((1-center_wing_linear_interpolation_coefficients)/2+0.5, plot=False).flatten()
    transition_chord_stretch_b_spline = lfs.Function(space=transition_parameters_b_spline_space, coefficients=transition_chord_stretch_coefficients)
    transition_chord_stretches = transition_chord_stretch_b_spline.evaluate(1-transition_linear_interpolation_coefficients).flatten()
    # transition_chord_stretches = csdl.linear_combination(transition_root_chord_stretch, transition_tip_chord_stretch, num_steps=grid_resolution[1],
    #                                                start_weights=transition_linear_interpolation_coefficients, stop_weights=1-transition_linear_interpolation_coefficients).flatten()

    wing_root_thickness_stretch = csdl.Variable(value=0.)
    wing_tip_thickness_stretch = csdl.Variable(value=0.)
    # center_wing_root_thickness_stretch = csdl.Variable(value=0.)
    # center_wing_tip_thickness_stretch = csdl.Variable(value=0.)
    center_wing_thickness_stretch_dofs = csdl.Variable(shape=(num_center_wing_spanwise_design_dofs,), value=0.)
    center_wing_thickness_stretch_coefficients = csdl.concatenate((center_wing_thickness_stretch_dofs[::-1], center_wing_thickness_stretch_dofs[1:]), axis=0)
    transition_root_thickness_stretch = center_wing_thickness_stretch_coefficients[-1]
    transition_tip_thickness_stretch = wing_root_thickness_stretch
    transition_thickness_stretch_coefficients = csdl.concatenate((transition_root_thickness_stretch.reshape(1,), transition_root_thickness_stretch.reshape(1,), transition_tip_thickness_stretch.reshape(1,), transition_tip_thickness_stretch.reshape(1,)), axis=0)

    wing_thickness_stretches = csdl.linear_combination(wing_root_thickness_stretch, wing_tip_thickness_stretch, num_steps=grid_resolution[1],
                                                    start_weights=wing_linear_interpolation_coefficients, stop_weights=1-wing_linear_interpolation_coefficients).flatten()
    # center_wing_thickness_stretches = csdl.linear_combination(center_wing_root_thickness_stretch, center_wing_tip_thickness_stretch, num_steps=grid_resolution[1],
    #                                                 start_weights=center_wing_linear_interpolation_coefficients, stop_weights=1-center_wing_linear_interpolation_coefficients).flatten()
    center_wing_thickness_stretches_b_spline = lfs.Function(space=center_wing_parameters_b_spline_space, coefficients=center_wing_thickness_stretch_coefficients.reshape((num_center_wing_spanwise_design_dofs*2-1,1)))
    center_wing_thickness_stretches = center_wing_thickness_stretches_b_spline.evaluate((1-center_wing_linear_interpolation_coefficients)/2+0.5, plot=False).flatten()
    # transition_thickness_stretches = csdl.linear_combination(transition_root_thickness_stretch, transition_tip_thickness_stretch, num_steps=grid_resolution[1],
    #                                                 start_weights=transition_linear_interpolation_coefficients, stop_weights=1-transition_linear_interpolation_coefficients).flatten()
    transition_thickness_stretches_b_spline = lfs.Function(space=transition_parameters_b_spline_space, coefficients=transition_thickness_stretch_coefficients)
    transition_thickness_stretches = transition_thickness_stretches_b_spline.evaluate(1-transition_linear_interpolation_coefficients, plot=False).flatten()

    if 'wing_twist_coefficients' in variables:
        wing_twist_coefficients = variables['wing_twist_coefficients']*np.pi/180
    else:
        wing_twist_coefficients = csdl.Variable(shape=(num_wing_twist_design_variables,), value=0.)
    if 'center_wing_twist_coefficients' in variables:
        center_wing_twist_dofs = variables['center_wing_twist_coefficients']*np.pi/180
    else:
        center_wing_twist_dofs = csdl.Variable(shape=(num_center_wing_spanwise_design_dofs,), value=0.)
    # wing_twist_coefficients = csdl.Variable(shape=(num_wing_twist_design_variables,), value=0.)
    wing_twist_b_spline = lfs.Function(space=wing_twist_b_spline_space, coefficients=wing_twist_coefficients)
    wing_twists = wing_twist_b_spline.evaluate(1-wing_linear_interpolation_coefficients, plot=False).flatten()
    # center_wing_twist_dofs = csdl.Variable(shape=(num_center_wing_spanwise_design_dofs,), value=0.)
    center_wing_twist_coefficients = csdl.concatenate((center_wing_twist_dofs[::-1], center_wing_twist_dofs[1:]), axis=0)
    center_wing_twist_b_spline = lfs.Function(space=center_wing_parameters_b_spline_space, coefficients=center_wing_twist_coefficients.reshape((num_center_wing_spanwise_design_dofs*2-1,1)))
    center_wing_twists = center_wing_twist_b_spline.evaluate((1-center_wing_linear_interpolation_coefficients)/2+0.5, plot=False).flatten()
    transition_twist_coefficients = csdl.concatenate((center_wing_twist_coefficients[-1].reshape(1,), center_wing_twist_coefficients[-1].reshape(1,), wing_twist_coefficients[0].reshape(1,), wing_twist_coefficients[0].reshape((1,))))
    transition_twist_b_spline = lfs.Function(space=transition_parameters_b_spline_space, coefficients=transition_twist_coefficients)
    transition_twists = transition_twist_b_spline.evaluate(1-transition_linear_interpolation_coefficients, plot=False).flatten() 

    right_wing_sectional_parameterization_inputs = lsdo_geo.VolumeSectionalParameterizationInputs(
        # translations={0: wing_sweep_translations, wingspan_stretch_axis: wing_wingspan_stretches, 2: wing_dihedral_translations},
        # stretches={0: wing_chord_stretches, 2: wing_thickness_stretches},
        # rotations={twist_axis: wing_twists}
    )
    if 'wing_sweep' in variables or 'transition_sweep' in variables:
        right_wing_sectional_parameterization_inputs.add_sectional_translation(0, wing_sweep_translations)
    if 'wing_half_span' in variables or 'transition_half_span' in variables or 'center_wing_half_span' in variables:
        right_wing_sectional_parameterization_inputs.add_sectional_translation(wingspan_stretch_axis, wing_wingspan_stretches)
    if 'wing_dihedral' in variables or 'transition_dihedral' in variables:
        right_wing_sectional_parameterization_inputs.add_sectional_translation(2, wing_dihedral_translations)
    if 'wing_root_chord' in variables or 'wing_tip_chord' in variables:
        right_wing_sectional_parameterization_inputs.add_sectional_stretch(0, wing_chord_stretches)
    if not enforce_thickness_to_chord_ratio:
        if 'wing_root_thickness' in variables or 'wing_tip_thickness' in variables:
            right_wing_sectional_parameterization_inputs.add_sectional_stretch(2, wing_thickness_stretches)
    else:
        if 'wing_root_chord' in variables or 'wing_tip_chord' in variables: # enforce t/c ratio
            right_wing_sectional_parameterization_inputs.add_sectional_stretch(2, wing_thickness_stretches)
    if 'wing_twist_coefficients' in variables:
        right_wing_sectional_parameterization_inputs.add_sectional_rotation(twist_axis, wing_twists)

    right_center_wing_sectional_parameterization_inputs = lsdo_geo.VolumeSectionalParameterizationInputs(
        # translations={0: center_wing_sweep_translations, wingspan_stretch_axis: center_wing_wingspan_stretches, 2: center_wing_dihedral_translations},
        # stretches={0: center_wing_chord_stretches, 2: center_wing_thickness_stretches},
        # rotations={twist_axis: center_wing_twists}
    )

    if not enforce_no_centerbody_trailing_edge_sweep:
        if 'center_wing_sweep' in variables:
            right_center_wing_sectional_parameterization_inputs.add_sectional_translation(0, center_wing_sweep_translations)
    else:
        if 'center_wing_chord_stretch_coefficients' in variables:
            right_center_wing_sectional_parameterization_inputs.add_sectional_translation(0, center_wing_sweep_translations)
    if 'center_wing_half_span' in variables:
        right_center_wing_sectional_parameterization_inputs.add_sectional_translation(wingspan_stretch_axis, center_wing_wingspan_stretches)
    if 'center_wing_dihedral' in variables:
        right_center_wing_sectional_parameterization_inputs.add_sectional_translation(2, center_wing_dihedral_translations)
    if 'center_wing_chord_stretch_coefficients' in variables:
        right_center_wing_sectional_parameterization_inputs.add_sectional_stretch(0, center_wing_chord_stretches)
    if not enforce_thickness_to_chord_ratio:
        if 'center_wing_root_thickness' in variables or 'center_wing_tip_thickness' in variables:
            right_center_wing_sectional_parameterization_inputs.add_sectional_stretch(2, center_wing_thickness_stretches)
    else:
        if 'center_wing_chord_stretch_coefficients' in variables: # enforce t/c ratio
            right_center_wing_sectional_parameterization_inputs.add_sectional_stretch(2, center_wing_thickness_stretches)
    if 'center_wing_twist_coefficients' in variables:
        right_center_wing_sectional_parameterization_inputs.add_sectional_rotation(twist_axis, center_wing_twists)

    right_transition_sectional_parameterization_inputs = lsdo_geo.VolumeSectionalParameterizationInputs(
        # translations={0: transition_sweep_translations, wingspan_stretch_axis: transition_wingspan_stretches, 2: transition_dihedral_translations},
        # stretches={0: transition_chord_stretches, 2: transition_thickness_stretches},
        # rotations={twist_axis: transition_twists}
    )

    if not enforce_no_centerbody_trailing_edge_sweep:
        if 'center_wing_sweep' in variables:
            right_transition_sectional_parameterization_inputs.add_sectional_translation(0, transition_sweep_translations)
    else:
        if 'center_wing_chord_stretch_coefficients' in variables:
            right_transition_sectional_parameterization_inputs.add_sectional_translation(0, transition_sweep_translations)
    if 'transition_half_span' in variables or 'center_wing_half_span' in variables:
        right_transition_sectional_parameterization_inputs.add_sectional_translation(wingspan_stretch_axis, transition_wingspan_stretches)
    if 'transition_dihedral' in variables or 'center_wing_dihedral' in variables:
        right_transition_sectional_parameterization_inputs.add_sectional_translation(2, transition_dihedral_translations)
    if 'wing_root_chord' in variables or 'center_wing_chord_stretch_coefficients' in variables:
        right_transition_sectional_parameterization_inputs.add_sectional_stretch(0, transition_chord_stretches)
    if not enforce_thickness_to_chord_ratio:
        if 'center_wing_tip_thickness' in variables or 'wing_root_thickness' in variables:
            right_transition_sectional_parameterization_inputs.add_sectional_stretch(2, transition_thickness_stretches)
    else:
        if 'wing_root_chord' in variables or 'center_wing_chord_stretch_coefficients' in variables: # enforce t/c ratio
            right_transition_sectional_parameterization_inputs.add_sectional_stretch(2, transition_thickness_stretches)
    if 'wing_twist_coefficients' in variables or 'center_wing_twist_coefficients' in variables:
        right_transition_sectional_parameterization_inputs.add_sectional_rotation(twist_axis, transition_twists)

    updated_right_wing_volume_mesh = right_wing_sectional_parameterization.evaluate(right_wing_sectional_parameterization_inputs, plot=False)
    updated_right_center_wing_volume_mesh = right_center_wing_sectional_parameterization.evaluate(right_center_wing_sectional_parameterization_inputs, plot=False)
    updated_right_transition_volume_mesh = right_transition_sectional_parameterization.evaluate(right_transition_sectional_parameterization_inputs, plot=False)

    # add elevator deflection as a local rotation on the last elevator_num_rows of discretization points for the center body
    elevator_num_rows = 5
    if 'elevator_deflection' in variables:
        elevator_deflection = variables['elevator_deflection']*np.pi/180
        elevator_rotation_axis = np.array([0., -1., 0.])
        elevator_rotation_point_upper = updated_right_center_wing_volume_mesh[-elevator_num_rows, 0, 1]
        elevator_rotation_point_lower = updated_right_center_wing_volume_mesh[-elevator_num_rows, 0, 0]
        elevator_rotation_point = (elevator_rotation_point_upper + elevator_rotation_point_lower) / 2
        elevator_points = updated_right_center_wing_volume_mesh[-elevator_num_rows:,:,:,:]
        elevator_points_shape = elevator_points.shape
        elevator_points_rotated = lsdo_geo.rotate(elevator_points, rotation_origin=elevator_rotation_point, axis_vector=elevator_rotation_axis, angles=elevator_deflection)
        updated_right_center_wing_volume_mesh = updated_right_center_wing_volume_mesh.set(csdl.slice[-elevator_num_rows:,:,:,:], elevator_points_rotated.reshape(elevator_points_shape))

    updated_left_wing_volume_mesh = updated_right_wing_volume_mesh[:,::-1,:,:]
    updated_left_wing_volume_mesh = updated_left_wing_volume_mesh.set(csdl.slice[:,:,:,1], -updated_left_wing_volume_mesh[:,:,:,1]) # flip y axis
    updated_left_center_wing_volume_mesh = updated_right_center_wing_volume_mesh[:,::-1,:,:]
    updated_left_center_wing_volume_mesh = updated_left_center_wing_volume_mesh.set(csdl.slice[:,:,:,1], -updated_left_center_wing_volume_mesh[:,:,:,1]) # flip y axis
    updated_left_transition_volume_mesh = updated_right_transition_volume_mesh[:,::-1,:,:]
    updated_left_transition_volume_mesh = updated_left_transition_volume_mesh.set(csdl.slice[:,:,:,1], -updated_left_transition_volume_mesh[:,:,:,1]) # flip y axis

    updated_lower_discretizations = lower_discretizations.copy()
    updated_lower_discretizations[-1] = (lower_discretizations[-1][0], updated_right_wing_volume_mesh[:,:,0,:])
    updated_lower_discretizations[-2] = (lower_discretizations[-2][0], updated_right_transition_volume_mesh[:,:,0,:])
    updated_lower_discretizations[-3] = (lower_discretizations[-3][0], updated_right_center_wing_volume_mesh[:,:,0,:])
    updated_lower_discretizations[-4] = (lower_discretizations[-4][0], updated_left_center_wing_volume_mesh[:,:,0,:])
    updated_lower_discretizations[-5] = (lower_discretizations[-5][0], updated_left_transition_volume_mesh[:,:,0,:])
    updated_lower_discretizations[-6] = (lower_discretizations[-6][0], updated_left_wing_volume_mesh[:,:,0,:])

    updated_upper_discretizations = upper_discretizations.copy()
    updated_upper_discretizations[-1] = (upper_discretizations[-1][0], updated_right_wing_volume_mesh[:,:,1,:])
    updated_upper_discretizations[-2] = (upper_discretizations[-2][0], updated_right_transition_volume_mesh[:,:,1,:])
    updated_upper_discretizations[-3] = (upper_discretizations[-3][0], updated_right_center_wing_volume_mesh[:,:,1,:])
    updated_upper_discretizations[-4] = (upper_discretizations[-4][0], updated_left_center_wing_volume_mesh[:,:,1,:])
    updated_upper_discretizations[-5] = (upper_discretizations[-5][0], updated_left_transition_volume_mesh[:,:,1,:])
    updated_upper_discretizations[-6] = (upper_discretizations[-6][0], updated_left_wing_volume_mesh[:,:,1,:])

    # stitch them together and refit to the new B-spline space
    lower_parametric_coordinates = []
    lower_points = []
    for discretization in updated_lower_discretizations:
        lower_parametric_coordinates.append(discretization[0])
        lower_points.append(discretization[1].reshape((-1, 3)))

    lower_parametric_coordinates = np.vstack(lower_parametric_coordinates)
    # lower_points = np.vstack(lower_points)
    lower_points = csdl.vstack(lower_points)

    upper_parametric_coordinates = []
    upper_points = []
    for discretization in updated_upper_discretizations:
        upper_parametric_coordinates.append(discretization[0])
        upper_points.append(discretization[1].reshape((-1, 3)))

    upper_parametric_coordinates = np.vstack(upper_parametric_coordinates)
    # upper_points = np.vstack(upper_points)
    upper_points = csdl.vstack(upper_points)

    # For repeated seam coordinates, keep one coordinate and average its point values.
    def deduplicate_and_average_points(parametric_coordinates, points):
        unique_coords, inverse = np.unique(parametric_coordinates, return_inverse=True, axis=0)
        counts = np.bincount(inverse)
        accumulated_points = np.zeros((unique_coords.shape[0], 3))
        np.add.at(accumulated_points, inverse, points)
        averaged_points = accumulated_points / counts[:, None]
        return unique_coords, averaged_points


    basis_matrix = oml_surface_b_spline_space.compute_basis_matrix(lower_parametric_coordinates)
    rhs = basis_matrix.T @ lower_points
    lower_oml_coefficients = (lower_inverted_fitting_matrix @ rhs).reshape((chordwise_resolution, num_coefficients_spanwise, 3))

    basis_matrix = oml_surface_b_spline_space.compute_basis_matrix(upper_parametric_coordinates)
    rhs = basis_matrix.T @ upper_points
    upper_oml_coefficients = (upper_inverted_fitting_matrix @ rhs).reshape((chordwise_resolution, num_coefficients_spanwise, 3))

    lower_oml = lfs.Function(space=oml_surface_b_spline_space, coefficients=lower_oml_coefficients)
    upper_oml = lfs.Function(space=oml_surface_b_spline_space, coefficients=upper_oml_coefficients)

    # Create wing caps using coefficients at each tip
    wing_cap_b_spline_space = lfs.BSplineSpaceNew(
        num_parametric_dimensions=2,
        degree=(3,1),
        coefficients_shape=(chordwise_resolution, 2)
    )
    left_wing_cap_coefficients_lower = lower_oml_coefficients[:, 0]
    left_wing_cap_coefficients_upper = upper_oml_coefficients[:, 0]
    left_wing_cap_coefficients = csdl.Variable(shape=(chordwise_resolution, 2, 3), value=0.)
    left_wing_cap_coefficients = left_wing_cap_coefficients.set(csdl.slice[:, 0, :], left_wing_cap_coefficients_lower)
    left_wing_cap_coefficients = left_wing_cap_coefficients.set(csdl.slice[:, 1, :], left_wing_cap_coefficients_upper)
    left_wing_cap = lfs.Function(space=wing_cap_b_spline_space, coefficients=left_wing_cap_coefficients)
    right_wing_cap_coefficients_lower = lower_oml_coefficients[:, -1]
    right_wing_cap_coefficients_upper = upper_oml_coefficients[:, -1]
    right_wing_cap_coefficients = csdl.Variable(shape=(chordwise_resolution, 2, 3), value=0.)
    right_wing_cap_coefficients = right_wing_cap_coefficients.set(csdl.slice[:, 0, :], right_wing_cap_coefficients_lower)
    right_wing_cap_coefficients = right_wing_cap_coefficients.set(csdl.slice[:, 1, :], right_wing_cap_coefficients_upper)
    right_wing_cap = lfs.Function(space=wing_cap_b_spline_space, coefficients=right_wing_cap_coefficients)

    new_geometry = lsdo_geo.Geometry(functions={0: lower_oml, 1: upper_oml, 2: left_wing_cap, 3: right_wing_cap})

    openvsp_geometry = geometry
    geometry = new_geometry

    # cumulative_sum_of_spanwise_resolutions = np.cumsum(spanwise_resolutions[:3]+[1]+spanwise_resolutions[3:]) # add 1 for the center section extra spanwise section
    # spanwise_section_boundaries = cumulative_sum_of_spanwise_resolutions / num_coefficients_spanwise
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
    center_wing_root_quarter_chord_upper = upper_oml.evaluate(np.array([0.667, 0.5]), plot=False).flatten()
    center_wing_root_quarter_chord_lower = lower_oml.evaluate(np.array([0.667, 0.5]), plot=False).flatten()
    center_wing_tip_quarter_chord_upper = upper_oml.evaluate(np.array([0.667, spanwise_section_boundaries[-3]]), plot=False).flatten()
    center_wing_tip_quarter_chord_lower = lower_oml.evaluate(np.array([0.667, spanwise_section_boundaries[-3]]), plot=False).flatten()
    center_wing_station_2_quarter_chord_upper = upper_oml.evaluate(np.array([0.667, 0.5 + (spanwise_section_boundaries[-3]-0.5)/3]), plot=False).flatten()
    center_wing_station_2_quarter_chord_lower = lower_oml.evaluate(np.array([0.667, 0.5 + (spanwise_section_boundaries[-3]-0.5)/3]), plot=False).flatten()
    center_wing_station_3_quarter_chord_upper = upper_oml.evaluate(np.array([0.667, 0.5 + 2*(spanwise_section_boundaries[-3]-0.5)/3]), plot=False).flatten()
    center_wing_station_3_quarter_chord_lower = lower_oml.evaluate(np.array([0.667, 0.5 + 2*(spanwise_section_boundaries[-3]-0.5)/3]), plot=False).flatten()

    wing_root_to_tip_vector = wing_tip - wing_root
    center_wing_root_to_tip_vector = center_wing_tip - center_wing_root
    transition_root_to_tip_vector = wing_root - center_wing_tip

    wing_sweep_computed = csdl.arctan2(wing_root_to_tip_vector[0], wing_root_to_tip_vector[1])
    # center_wing_sweep_computed = csdl.arctan2(center_wing_root_to_tip_vector[0], center_wing_root_to_tip_vector[1])
    # center_wing_trailing_edge_sweep_1 = csdl.arctan2(center_wing_station_2_trailing_edge[0]-center_wing_root_trailing_edge[0], center_wing_station_2_trailing_edge[1]-center_wing_root_trailing_edge[1])
    # center_wing_trailing_edge_sweep_2 = csdl.arctan2(center_wing_station_3_trailing_edge[0]-center_wing_root_trailing_edge[0], center_wing_station_3_trailing_edge[1]-center_wing_root_trailing_edge[1])
    # center_wing_trailing_edge_sweep_3 = csdl.arctan2(center_wing_tip_trailing_edge[0]-center_wing_root_trailing_edge[0], center_wing_tip_trailing_edge[1]-center_wing_root_trailing_edge[1])
    # NOTE: Although the above computations are correct, it seems more efficient to just enforce that the change in x must be 0.
    center_wing_trailing_edge_sweep_1_constraint = center_wing_station_2_trailing_edge[0] - center_wing_root_trailing_edge[0]
    center_wing_trailing_edge_sweep_2_constraint = center_wing_station_3_trailing_edge[0] - center_wing_root_trailing_edge[0]
    center_wing_trailing_edge_sweep_3_constraint = center_wing_tip_trailing_edge[0] - center_wing_root_trailing_edge[0]
    center_wing_trailing_edge_constraints = csdl.concatenate((center_wing_trailing_edge_sweep_1_constraint, center_wing_trailing_edge_sweep_2_constraint, center_wing_trailing_edge_sweep_3_constraint), axis=0)
    transition_sweep_computed = csdl.arctan2(transition_root_to_tip_vector[0], transition_root_to_tip_vector[1])

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
    # center_wing_root_thickness_computed = upper_center_wing_root[2] - lower_center_wing_root[2]
    # center_wing_tip_thickness_computed = upper_center_wing_tip[2] - lower_center_wing_tip[2]
    wing_root_thickness_to_chord_ratio_computed = wing_root_thickness_computed / wing_root_chord_computed
    wing_tip_thickness_to_chord_ratio_computed = wing_tip_thickness_computed / wing_tip_chord_computed

    center_wing_root_thickness_computed = center_wing_root_quarter_chord_upper[2] - center_wing_root_quarter_chord_lower[2]
    center_wing_tip_thickness_computed = center_wing_tip_quarter_chord_upper[2] - center_wing_tip_quarter_chord_lower[2]
    center_wing_station_2_thickness_computed = center_wing_station_2_quarter_chord_upper[2] - center_wing_station_2_quarter_chord_lower[2]
    center_wing_station_3_thickness_computed = center_wing_station_3_quarter_chord_upper[2] - center_wing_station_3_quarter_chord_lower[2]
    center_wing_root_thickness_to_chord_ratio_computed = center_wing_root_thickness_computed / center_wing_root_chord_computed
    center_wing_tip_thickness_to_chord_ratio_computed = center_wing_tip_thickness_computed / center_wing_tip_chord_computed
    center_wing_station_2_thickness_to_chord_ratio_computed = center_wing_station_2_thickness_computed / center_wing_station_2_chord_computed
    center_wing_station_3_thickness_to_chord_ratio_computed = center_wing_station_3_thickness_computed / center_wing_station_3_chord_computed
    center_wing_thickness_to_chord_ratios_computed = csdl.concatenate((center_wing_root_thickness_to_chord_ratio_computed.reshape(1,),
                                                                    center_wing_station_2_thickness_to_chord_ratio_computed.reshape(1,), 
                                                                    center_wing_station_3_thickness_to_chord_ratio_computed.reshape(1,), 
                                                                    center_wing_tip_thickness_to_chord_ratio_computed.reshape(1,)), axis=0)

    # enforce geometric constraints and add geometric variables to the solver
    if enforce_interface_smoothing:
        # transition_center_wing_interface_leading_edge_curvature = upper_oml.evaluate(np.array([[0.0, spanwise_section_boundaries[-3]]]), parametric_derivative_orders=(0,2))
        # transition_center_wing_interface_trailing_edge_curvature = upper_oml.evaluate(np.array([[1.0, spanwise_section_boundaries[-3]]]), parametric_derivative_orders=(0,2))
        # transition_wing_interface_leading_edge_curvature = upper_oml.evaluate(np.array([[0.0, spanwise_section_boundaries[-2]]]), parametric_derivative_orders=(0,2))
        # transition_wing_interface_trailing_edge_curvature = upper_oml.evaluate(np.array([[1.0, spanwise_section_boundaries[-2]]]), parametric_derivative_orders=(0,2))
        num_transition_trailing_edge_sample_points = 21
        transition_leading_edge_parametric_coordinates = np.linspace(np.array([0.0, spanwise_section_boundaries[-3]]), np.array([0.0, spanwise_section_boundaries[-2]-0.003]), num_transition_trailing_edge_sample_points)
        transition_trailing_edge_parametric_coordinates = np.linspace(np.array([1.0, spanwise_section_boundaries[-3]]), np.array([1.0, spanwise_section_boundaries[-2]-0.003]), num_transition_trailing_edge_sample_points)
        # transition_trailing_edge_parametric_coordinates = np.cos(np.array([1.0, spanwise_section_boundaries[-3]]), np.array([1.0, spanwise_section_boundaries[-2]]), num_transition_trailing_edge_sample_points)
        # transition_trailing_edge_parametric_coordinates_v = (np.sin(np.linspace(-np.pi/2, np.pi/2, num_transition_trailing_edge_sample_points))+1.)/2*(spanwise_section_boundaries[-2]-(spanwise_section_boundaries[-3]-0.01)) + spanwise_section_boundaries[-3]-0.01
        # transition_trailing_edge_parametric_coordinates = np.hstack((np.ones((num_transition_trailing_edge_sample_points,1)), transition_trailing_edge_parametric_coordinates_v.reshape(-1,1)))
        upper_oml.evaluate(transition_leading_edge_parametric_coordinates, plot=False)
        upper_oml.evaluate(transition_trailing_edge_parametric_coordinates, plot=False)
        transition_leading_edge_curvatures = upper_oml.evaluate(transition_leading_edge_parametric_coordinates, parametric_derivative_orders=(0,2))
        transition_trailing_edge_curvatures = upper_oml.evaluate(transition_trailing_edge_parametric_coordinates, parametric_derivative_orders=(0,2))
        # penalty_factors = np.cos(np.linspace(0, np.pi, num_transition_trailing_edge_sample_points))**2*0.99 + 0.01
        penalty_factors = np.cos(np.linspace(0, np.pi, num_transition_trailing_edge_sample_points))**4
        penalty_factors[0] *= 5.
        # penalty_factors = np.ones((num_transition_trailing_edge_sample_points,))
        # penalty_factors*=1.e-4
        # transition_curvatures = csdl.concatenate((transition_center_wing_interface_leading_edge_curvature, transition_center_wing_interface_trailing_edge_curvature,
        #                                         transition_wing_interface_leading_edge_curvature, transition_wing_interface_trailing_edge_curvature), axis=0)
        # normalized_transition_curvatures = transition_curvatures / transition_curvatures.value - 1.
        # curvature_penalties = csdl.sum(normalized_transition_curvatures**2)
        # parameterization_solver.add_equality_constraint(curvature_penalties, 0., penalty=1.e-2)
        # parameterization_solver.add_equality_constraint(transition_center_wing_interface_trailing_edge_curvature[0], 0., penalty=1.e-5)
        # parameterization_solver.add_equality_constraint(transition_center_wing_interface_trailing_edge_curvature[0], 0., penalty=1.e-4)
        
        parameterization_solver.add_equality_constraint(transition_leading_edge_curvatures[:,0]*penalty_factors, 0., penalty=1.e-5)
        parameterization_solver.add_equality_constraint(transition_trailing_edge_curvatures[:,0]*penalty_factors, 0., penalty=1.e-5)
        parameterization_solver.add_state(transition_stretch_internal_coefficients, cost=1.e-2)


    if enforce_no_centerbody_trailing_edge_sweep:
        parameterization_solver.add_equality_constraint(center_wing_trailing_edge_constraints, 0.)
        parameterization_solver.add_state(center_wing_sweep_translation_dofs, cost=1.e-2)
        # added 4 dofs and only 3 constraints so add an x location anchor to prevent drifting in x direction
        anchor_point = upper_oml.evaluate(np.array([[0.667, 0.5]]), plot=False).flatten()
        parameterization_solver.add_equality_constraint(anchor_point[0], anchor_point[0].value)


    if enforce_thickness_to_chord_ratio:
        if 'wing_root_chord' in variables.keys():
            wing_root_thickness_to_chord_ratio = wing_root_thickness_to_chord_ratio_computed.value
            parameterization_solver.add_equality_constraint(wing_root_thickness_to_chord_ratio_computed, wing_root_thickness_to_chord_ratio)
            parameterization_solver.add_state(wing_root_thickness_stretch)
        if 'wing_tip_chord' in variables.keys():
            wing_tip_thickness_to_chord_ratio = wing_tip_thickness_to_chord_ratio_computed.value
            parameterization_solver.add_equality_constraint(wing_tip_thickness_to_chord_ratio_computed, wing_tip_thickness_to_chord_ratio)
            parameterization_solver.add_state(wing_tip_thickness_stretch)
        if 'center_wing_chord_stretch_coefficients' in variables.keys():
            center_wing_thickness_to_chord_ratios = center_wing_thickness_to_chord_ratios_computed.value
            parameterization_solver.add_equality_constraint(center_wing_thickness_to_chord_ratios_computed, center_wing_thickness_to_chord_ratios)
            parameterization_solver.add_state(center_wing_thickness_stretch_dofs)

    
    geometric_variables = lsdo_geo.GeometricVariables()
    if 'wing_sweep' in variables.keys():
        wing_sweep = variables['wing_sweep']*np.pi/180
        geometric_variables.add_variable(computed_value=wing_root_to_tip_vector[0] / wing_root_to_tip_vector[1], desired_value=csdl.tan(wing_sweep))
        parameterization_solver.add_state(wing_tip_sweep_translation)
    if 'center_wing_sweep' in variables.keys():
        center_wing_sweep = variables['center_wing_sweep']*np.pi/180
        geometric_variables.add_variable(computed_value=center_wing_root_to_tip_vector[0] / center_wing_root_to_tip_vector[1], desired_value=csdl.tan(center_wing_sweep))
        parameterization_solver.add_state(center_wing_tip_sweep_translation)
    if 'transition_sweep' in variables.keys():
        transition_sweep = variables['transition_sweep']*np.pi/180
        geometric_variables.add_variable(computed_value=transition_root_to_tip_vector[0] / transition_root_to_tip_vector[1], desired_value=csdl.tan(transition_sweep))
        parameterization_solver.add_state(transition_tip_sweep_translation)
    if 'wing_half_span' in variables.keys():
        wing_half_span = variables['wing_half_span']
        geometric_variables.add_variable(computed_value=wing_half_span_computed, desired_value=wing_half_span)
        parameterization_solver.add_state(wing_tip_wingspan_stretch)
    if 'center_wing_half_span' in variables.keys():
        center_wing_half_span = variables['center_wing_half_span']
        geometric_variables.add_variable(computed_value=center_wing_half_span_computed, desired_value=center_wing_half_span)
        parameterization_solver.add_state(center_wing_tip_wingspan_stretch)
    if 'transition_half_span' in variables.keys():
        transition_half_span = variables['transition_half_span']
        geometric_variables.add_variable(computed_value=transition_half_span_computed, desired_value=transition_half_span)
        parameterization_solver.add_state(transition_tip_wingspan_stretch)
    if 'wing_dihedral' in variables.keys():
        wing_dihedral = variables['wing_dihedral']*np.pi/180
        geometric_variables.add_variable(computed_value=wing_root_to_tip_vector[2] / wing_root_to_tip_vector[1], desired_value=csdl.tan(wing_dihedral))
        parameterization_solver.add_state(wing_tip_dihedral_translation)
    if 'center_wing_dihedral' in variables.keys():
        center_wing_dihedral = variables['center_wing_dihedral']*np.pi/180
        geometric_variables.add_variable(computed_value=center_wing_root_to_tip_vector[2] / center_wing_root_to_tip_vector[1], desired_value=csdl.tan(center_wing_dihedral))
        parameterization_solver.add_state(center_wing_tip_dihedral_translation)
    if 'transition_dihedral' in variables.keys():
        transition_dihedral = variables['transition_dihedral']*np.pi/180
        geometric_variables.add_variable(computed_value=transition_root_to_tip_vector[2] / transition_root_to_tip_vector[1], desired_value=csdl.tan(transition_dihedral))
        parameterization_solver.add_state(transition_tip_dihedral_translation)
    if 'wing_root_chord' in variables.keys():
        wing_root_chord = variables['wing_root_chord']
        geometric_variables.add_variable(computed_value=wing_root_chord_computed, desired_value=wing_root_chord)
        parameterization_solver.add_state(wing_root_chord_stretch)
    if 'wing_tip_chord' in variables.keys():
        wing_tip_chord = variables['wing_tip_chord']
        geometric_variables.add_variable(computed_value=wing_tip_chord_computed, desired_value=wing_tip_chord)
        parameterization_solver.add_state(wing_tip_chord_stretch)
    if 'center_wing_root_chord' in variables.keys():
        center_wing_root_chord = variables['center_wing_root_chord']
        geometric_variables.add_variable(computed_value=center_wing_root_chord_computed, desired_value=center_wing_root_chord)
        parameterization_solver.add_state(center_wing_root_chord_stretch)
    if 'center_wing_tip_chord' in variables.keys():
        center_wing_tip_chord = variables['center_wing_tip_chord']
        geometric_variables.add_variable(computed_value=center_wing_tip_chord_computed, desired_value=center_wing_tip_chord)
        parameterization_solver.add_state(center_wing_tip_chord_stretch)
    if 'center_wing_chords' in variables.keys():
        center_wing_chords = variables['center_wing_chords']
        geometric_variables.add_variable(computed_value=center_wing_chords_computed, desired_value=center_wing_chords)
        parameterization_solver.add_state(center_wing_chord_stretch_dofs)
    
    if not enforce_thickness_to_chord_ratio:
        if 'wing_root_thickness_to_chord_ratio' in variables.keys():
            wing_root_thickness_to_chord_ratio = variables['wing_root_thickness_to_chord_ratio']
            geometric_variables.add_variable(computed_value=wing_root_thickness_to_chord_ratio_computed, desired_value=wing_root_thickness_to_chord_ratio)
            parameterization_solver.add_state(wing_root_thickness_stretch)
        if 'wing_tip_thickness_to_chord_ratio' in variables.keys():
            wing_tip_thickness_to_chord_ratio = variables['wing_tip_thickness_to_chord_ratio']
            geometric_variables.add_variable(computed_value=wing_tip_thickness_to_chord_ratio_computed, desired_value=wing_tip_thickness_to_chord_ratio)
            parameterization_solver.add_state(wing_tip_thickness_stretch)        
        if 'center_wing_thickness_to_chord_ratios' in variables.keys():
            center_wing_thickness_to_chord_ratios = variables['center_wing_thickness_to_chord_ratios']
            geometric_variables.add_variable(computed_value=center_wing_thickness_to_chord_ratios_computed, desired_value=center_wing_thickness_to_chord_ratios)
            parameterization_solver.add_state(center_wing_thickness_stretch_dofs)
    
    if len(geometric_variables.desired_value) > 0:
        parameterization_solver.evaluate(geometric_variables)

    if plot:
        openvsp_geometry_plot = openvsp_geometry.plot(opacity=0.7, show=False)
        geometry.plot(show=True, color='red', opacity=0.7, additional_plotting_elements=[openvsp_geometry_plot])

    return geometry


