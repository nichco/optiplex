import csdl_alpha as csdl
import numpy as np
# from talos.utils.bspline_comp import BsplineComp, get_bspline_mtx
# from talos.disciplines.reference_frames.body123 import body123_reference_frame_change
# from talos.disciplines.attitude.orbit_body_reference_frame import orbit_body_reference_frame_change
 
import numpy as np
import scipy.sparse
import csdl_alpha as csdl
 
 
def get_bspline_mtx(num_cp, num_pt, order=4):
    order = min(order, num_cp)
    knots = np.zeros(num_cp + order)
    knots[order - 1:num_cp + 1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp + 1:] = 1.0
    t_vec = np.linspace(0, 1, num_pt)
    basis = np.zeros(order)
    arange = np.arange(order)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)
    for ipt in range(num_pt):
        t = t_vec[ipt]
        i0 = -1
        for ind in range(order, num_cp + 1):
            if (knots[ind - 1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order
        basis[:] = 0.
        basis[-1] = 1.
        for i in range(2, order + 1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n + l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1 - 1] = 0.
            for j in range(j1 + 1, j2):
                n = i0 + j
                if knots[n + l - 1] != knots[n - 1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j - 1] = 0.
                if knots[n + l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n + l - 1] != knots[n - 1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2 - 1] = 0.
        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange
    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()
    return scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_pt, num_cp),
    )
 
 
class BsplineComp(csdl.CustomExplicitOperation):
    """
    Translates control points to actual points using a B-spline.
    """
    def __init__(self, num_cp, num_pt, jac, in_name, out_name):
        super().__init__()
        self.num_cp = num_cp
        self.num_pt = num_pt
        self.jac = jac
        self.in_name = in_name
        self.out_name = out_name
 
    def evaluate(self, inputs: csdl.VariableGroup):
        self.declare_input(self.in_name, getattr(inputs, self.in_name))
        output = self.create_output(self.out_name, shape=(self.num_pt,))
        self.declare_derivative_parameters(self.out_name, self.in_name, dependent=True)
        out = csdl.VariableGroup()
        setattr(out, self.out_name, output)
        return out
 
    def compute(self, input_vals, output_vals):
        output_vals[self.out_name] = self.jac @ input_vals[self.in_name]
 
    def compute_derivatives(self, input_vals, output_vals, derivatives):
        derivatives[self.out_name, self.in_name] = self.jac.toarray().astype(float)
 
 
 
 
import csdl_alpha as csdl
import numpy as np
# Update for new CSDL version, using csdl.Variable instead of self.declare_variable, and using csdl.slice to build up the output variable slot by slot, since CSDL needs to track each assignment to the variable
def body123_reference_frame_change(yaw, pitch, roll, num_times):
    # Computes 123 rotation matrix C, (shape 3, 3, num_times) describing how to express coordinates in the body frame that were originally defined in the ECI (Earth Centered Inertial) frame, given time histories of yaw, pitch, and roll angles (shape (num_times, ))
    sr = csdl.sin(roll)
    cr = csdl.cos(roll)
    sp = csdl.sin(pitch)
    cp = csdl.cos(pitch)
    sy = csdl.sin(yaw)
    cy = csdl.cos(yaw)
 
    # Create a tracked output variable and fill it slot-by-slot using csdl.slice.
    # New CSDL will broadcast the (num_times,) arrays automatically, so no csdl.expand is needed.
    C = csdl.Variable(value=np.zeros((3, 3, num_times)))
    C = C.set(csdl.slice[0, 0, :], cp * cy)
    C = C.set(csdl.slice[0, 1, :], cp * sy)
    C = C.set(csdl.slice[0, 2, :], -sp)
    C = C.set(csdl.slice[1, 0, :], sr * sp * cy - cr * sy)
    C = C.set(csdl.slice[1, 1, :], sr * sp * sy + cr * cy)
    C = C.set(csdl.slice[1, 2, :], cp * sr)
    C = C.set(csdl.slice[2, 0, :], cr * sp * cy + sr * sy)
    C = C.set(csdl.slice[2, 1, :], cr * sp * sy - sr * cy)
    C = C.set(csdl.slice[2, 2, :], cp * cr)
 
    return C
# Derivative check and test of the body123_reference_frame_change function, using csdl.Recorder to track operations and check derivatives with respect to the input variables yaw, pitch, and roll, and printing the value of C at t=0 to verify it is the identity matrix when all angles are zero
if __name__ == "__main__":
    num_times = 100
    recorder = csdl.Recorder(inline=True)
    recorder.start()
 
    yaw   = csdl.Variable(value=np.zeros(num_times), name='yaw')
    pitch = csdl.Variable(value=np.zeros(num_times), name='pitch')
    roll  = csdl.Variable(value=np.zeros(num_times), name='roll')
 
    C = body123_reference_frame_change(yaw, pitch, roll, num_times)
 
    recorder.stop()
    print("C at t=0 (should be identity):\n", C.value[:, :, 0])
 
 
# Attitude.py defines an attitude optimization problem for a spacecraft in orbit, using reaction wheels for control, includes effects of gravity gradient torque, integrated in time, and includes constraints on reaction wheel torque and speed, with design variables for the attitude trajectory and initial reaction wheel speeds, and uses CSDL for automatic differentiation and optimization
def orbit_body_reference_frame_change(RTN_from_ECI, B_from_ECI, num_times, step_size):
        ECI_from_RTN = csdl.reorder_axes(RTN_from_ECI, 'ijk->jik')
        B_from_RTN = csdl.einsum(B_from_ECI, ECI_from_RTN, action='ijl,jkl->ikl')
        # Rate of change of Reference frame transformation
        B_from_ECI_dot = csdl.Variable(value = np.zeros((3, 3, num_times)))
        # Next - current time step, divided by step_size gives rate of change
        B_from_ECI_dot = B_from_ECI_dot.set(csdl.slice[:, :, 1:], (B_from_ECI[:, :, 1:] - B_from_ECI[:, :, :-1]) / step_size
        )
        return B_from_RTN, B_from_ECI_dot
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
def body_rates(B_from_ECI, B_from_ECI_dot, osculating_orbit_angular_speed,
               sc_mmoi, step_size, num_times, gravity_gradient, B_from_RTN=None):
    # Angular velocity via skew-symmetric cross operator: wcross = B_dot @ B^T
    wcross = csdl.einsum(
        B_from_ECI_dot,
        csdl.einsum(B_from_ECI, action='ijk->jik'),
        action='ijl,jkl->ikl')
 
    # Extract angular velocity components from skew-symmetric matrix
    rates = csdl.Variable(value=np.zeros((num_times, 3)))
    rates = rates.set(csdl.slice[:, 0], wcross[2, 1, :])
    rates = rates.set(csdl.slice[:, 1], wcross[0, 2, :])
    rates = rates.set(csdl.slice[:, 2], wcross[1, 0, :])
 
    # Angular acceleration via finite differences
    accels = csdl.Variable(value=np.zeros((num_times, 3)))
    accels = accels.set(
        csdl.slice[1:, :],
        (rates[1:, :] - rates[:-1, :]) / step_size
    )
 
    # Jw = angular momentum = J * omega
    Jw = rates * np.einsum('i,j->ij', np.ones(num_times), sc_mmoi)
 
    # bt1 = J * alpha (inertial torque)
    bt1 = csdl.Variable(value=np.zeros((num_times, 3)))
    bt1 = bt1.set(csdl.slice[:, 0], sc_mmoi[0] * accels[:, 0])
    bt1 = bt1.set(csdl.slice[:, 1], sc_mmoi[1] * accels[:, 1])
    bt1 = bt1.set(csdl.slice[:, 2], sc_mmoi[2] * accels[:, 2])
 
    # bt2 = omega x (J * omega) (gyroscopic torque)
    bt2 = csdl.cross(rates, Jw, axis=1)
 
    # B_from_RTN is the rotation matrix that transforms vectors from the RTN to Body frame
    # RTN - Radial points away from Earth, Tangential points in the direction of motion along orbit, Normal points perpendicularly to the orbital plane
    # Gravity gradient torque arises from the non-uniform gravity field of Earth
    # T = -3 * (I2 - I3) * B_from_RTN[1,0] * B_from_RTN[2,0] * n^2 for x-axis
    # where n is the orbital angular velocity, with cyclic permutations for y and z axes
    if gravity_gradient is True:
        bt3 = csdl.Variable(value=np.zeros((3, num_times)))
        bt3 = bt3.set(csdl.slice[0, :],
            -3 * (sc_mmoi[1] - sc_mmoi[2]) * B_from_RTN[1, 0, :] * B_from_RTN[2, 0, :] * osculating_orbit_angular_speed[0, :]**2)
        bt3 = bt3.set(csdl.slice[1, :],
            -3 * (sc_mmoi[2] - sc_mmoi[0]) * B_from_RTN[2, 0, :] * B_from_RTN[0, 0, :] * osculating_orbit_angular_speed[0, :]**2)
        bt3 = bt3.set(csdl.slice[2, :],
            -3 * (sc_mmoi[0] - sc_mmoi[1]) * B_from_RTN[0, 0, :] * B_from_RTN[1, 0, :] * osculating_orbit_angular_speed[0, :]**2)
        body_torque = bt1 + bt2 + csdl.reorder_axes(bt3, 'ij->ji')
    else:
        body_torque = bt1 + bt2
 
    return rates, body_torque
 
 
def reaction_wheel_dynamics(omega, body_rates, body_torque, rw_mmoi):
    # x = J_rw * omega_body (angular momentum of reaction wheels)
    x = csdl.Variable(value=np.zeros(3))
    x = x.set(csdl.slice[0], rw_mmoi[0] * body_rates[0])
    x = x.set(csdl.slice[1], rw_mmoi[1] * body_rates[1])
    x = x.set(csdl.slice[2], rw_mmoi[2] * body_rates[2])
    # dw_dt = omega_body x (J_rw * omega_body) - body_torque
    dw_dt = csdl.cross(body_rates, x, axis=0) - body_torque
    return dw_dt
 
 
def runge_kutta_4(f, omega0, body_rates_history, body_torque_history, h, n, rw_mmoi):
    # Preallocate history array and set initial condition
    omega = omega0
    omega_history = csdl.Variable(value=np.zeros((n + 1, 3)))
    omega_history = omega_history.set(csdl.slice[0, :], omega0)
 
    for i in csdl.frange(n):
        # Look up spacecraft angular velocity and torque at this time step
        B = body_rates_history[i, :]
        torque = body_torque_history[i, :]
 
        # RK4 derivative estimates
        k1 = f(omega, B, torque, rw_mmoi)
        k2 = f(omega + 0.5*h*k1, B, torque, rw_mmoi)
        k3 = f(omega + 0.5*h*k2, B, torque, rw_mmoi)
        k4 = f(omega + h*k3, B, torque, rw_mmoi)
 
        # Weighted average update
        omega = omega + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        omega_history = omega_history.set(csdl.slice[i+1, :], omega)
 
    return omega_history
 
 
# def attitude(num_times, num_cp, step_size, RTN_from_ECI, osculating_orbit_angular_speed,
#              max_rw_torque=0.004,
#              sc_mmoi=6 * np.array([2, 1, 3]) * 1e-3,
#              rw_mmoi=6 * np.ones(3) * 1e-5,
#              gravity_gradient=True):
 
#     if sc_mmoi.shape != (3,):
#         raise ValueError('sc_mmoi must have shape (3,); has shape {}'.format(sc_mmoi.shape))
#     if rw_mmoi.shape != (3,):
#         raise ValueError('rw_mmoi must have shape (3,); has shape {}'.format(rw_mmoi.shape))
 
#     max_rw_speed = 1.0 / max_rw_torque
 
#     # B-spline control points (design variables)
#     jac = get_bspline_mtx(num_cp, num_times)
#     # setting values to test
#     yaw_cp = csdl.Variable(value=np.linspace(0, 0.1, num_cp), name='yaw_cp')
#     pitch_cp = csdl.Variable(value=np.linspace(0, 0.05, num_cp), name='pitch_cp')
#     roll_cp = csdl.Variable(value=np.linspace(0, 0.05, num_cp), name='roll_cp')
#     # yaw_cp = csdl.Variable(value=np.zeros(num_cp), name='yaw_cp')
#     # pitch_cp = csdl.Variable(value=np.zeros(num_cp), name='pitch_cp')
#     # roll_cp = csdl.Variable(value=np.zeros(num_cp), name='roll_cp')
#     # scale these
#     yaw_cp.set_as_design_variable(scaler=1e10)
#     pitch_cp.set_as_design_variable(scaler=1e10)
#     roll_cp.set_as_design_variable(scaler=1e10)
 
#     yaw_inputs = csdl.VariableGroup()
#     yaw_inputs.yaw_cp = yaw_cp
#     yaw = BsplineComp(num_cp=num_cp, num_pt=num_times, jac=jac,
#                       in_name='yaw_cp', out_name='yaw').evaluate(yaw_inputs).yaw
 
#     pitch_inputs = csdl.VariableGroup()
#     pitch_inputs.pitch_cp = pitch_cp
#     pitch = BsplineComp(num_cp=num_cp, num_pt=num_times, jac=jac,
#                         in_name='pitch_cp', out_name='pitch').evaluate(pitch_inputs).pitch
 
#     roll_inputs = csdl.VariableGroup()
#     roll_inputs.roll_cp = roll_cp
#     roll = BsplineComp(num_cp=num_cp, num_pt=num_times, jac=jac,
#                        in_name='roll_cp', out_name='roll').evaluate(roll_inputs).roll
#     # set final constraints for attitude to force optimization problem towards final target orientation minimizing torque along the way
#     # anytime you add a constraint add a scaler to make it order of magnitude 1
#     #yaw[-1].set_as_constraint(equals=0.1, scaler=10.0)
#     #roll[-1].set_as_constraint(equals=0.05, scaler=20.0)
#     #pitch[-1].set_as_constraint(equals=0.05, scaler=20.0)
#     yaw[-1].set_as_constraint(equals=0)
#     roll[-1].set_as_constraint(equals=0)
#     pitch[-1].set_as_constraint(equals=0)
#     # Reference frame transformations
#     B_from_ECI = body123_reference_frame_change(yaw, pitch, roll, num_times)
#     B_from_RTN, B_from_ECI_dot = orbit_body_reference_frame_change(RTN_from_ECI, B_from_ECI, num_times, step_size)
#     rates, body_torque = body_rates(B_from_ECI, B_from_ECI_dot, osculating_orbit_angular_speed,
#                                     sc_mmoi, step_size, num_times, gravity_gradient, B_from_RTN)
 
#     # Initial reaction wheel velocity (design variable)
#     initial_rw_velocity = csdl.Variable(value=np.zeros(3), name='initial_reaction_wheel_velocity')
#    # initial_rw_velocity.set_as_design_variable(lower=-max_rw_speed, upper=max_rw_speed)
 
#     # RK4 integration of reaction wheel dynamics
#     rw_velocity_history = runge_kutta_4(
#         reaction_wheel_dynamics,
#         initial_rw_velocity,
#         rates,
#         body_torque,
#         step_size,
#         num_times - 1,
#         rw_mmoi
#     )
 
#     # Reaction wheel acceleration and torque
#     rw_accel_history = csdl.Variable(value=np.zeros((num_times, 3)))
#     rw_accel_history = rw_accel_history.set(
#         csdl.slice[1:, :],
#         (rw_velocity_history[1:, :] - rw_velocity_history[:-1, :]) / step_size
#     )
#     reaction_wheel_torque = csdl.Variable(value=np.zeros((num_times, 3)))
#     reaction_wheel_torque = reaction_wheel_torque.set(csdl.slice[:, 0], rw_mmoi[0] * rw_accel_history[:, 0])
#     reaction_wheel_torque = reaction_wheel_torque.set(csdl.slice[:, 1], rw_mmoi[1] * rw_accel_history[:, 1])
#     reaction_wheel_torque = reaction_wheel_torque.set(csdl.slice[:, 2], rw_mmoi[2] * rw_accel_history[:, 2])
 
#     # Constraints on reaction wheel torque
#     # min and max need to be adjusted
#     min_rw_torque = csdl.minimum(reaction_wheel_torque)
#     max_rw_torque_val = csdl.maximum(reaction_wheel_torque)
#     min_rw_torque.set_as_constraint(lower=-max_rw_torque, scaler=1e150)
#     max_rw_torque_val.set_as_constraint(upper=max_rw_torque, scaler=1e150)
#     rw_effort = csdl.sum(reaction_wheel_torque ** 2)
#     rw_effort.set_as_objective(scaler=1e20)
#     return rw_velocity_history, reaction_wheel_torque, yaw, pitch, roll
 




def attitude(
    num_times,
    num_cp,
    step_size,
    RTN_from_ECI,
    osculating_orbit_angular_speed,
    max_rw_torque=0.004,
    sc_mmoi=6 * np.array([2, 1, 3]) * 1e-3,
    rw_mmoi=6 * np.ones(3) * 1e-5,
    gravity_gradient=True,
    initial_yaw=0.1,
    initial_pitch=0.05,
    initial_roll=0.05,
):
    """
    Minimum-reaction-wheel-effort attitude slew.

    Initial attitude:
        (initial_yaw, initial_pitch, initial_roll)

    Final attitude:
        (0,0,0)

    Additional physical requirements:
        - wheel speeds start at zero
        - wheel speeds end at zero
        - wheel torque limits enforced
        - wheel speed limits enforced

    Objective:
        minimize integrated wheel torque squared
        + attitude smoothness regularization
    """

    if sc_mmoi.shape != (3,):
        raise ValueError(
            f"sc_mmoi must have shape (3,), got {sc_mmoi.shape}"
        )

    if rw_mmoi.shape != (3,):
        raise ValueError(
            f"rw_mmoi must have shape (3,), got {rw_mmoi.shape}"
        )

    max_rw_speed = 1.0 / max_rw_torque

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    yaw_scaler = 1.0 / max(abs(initial_yaw), 1e-6)
    pitch_scaler = 1.0 / max(abs(initial_pitch), 1e-6)
    roll_scaler = 1.0 / max(abs(initial_roll), 1e-6)

    torque_scaler = 1.0 / max_rw_torque
    speed_scaler = 1.0 / max_rw_speed

    # ------------------------------------------------------------------
    # B-spline parameterization
    # ------------------------------------------------------------------

    jac = get_bspline_mtx(num_cp, num_times)

    yaw_cp = csdl.Variable(
        value=np.linspace(initial_yaw, 0.0, num_cp),
        name="yaw_cp",
    )

    pitch_cp = csdl.Variable(
        value=np.linspace(initial_pitch, 0.0, num_cp),
        name="pitch_cp",
    )

    roll_cp = csdl.Variable(
        value=np.linspace(initial_roll, 0.0, num_cp),
        name="roll_cp",
    )

    yaw_cp.set_as_design_variable(scaler=yaw_scaler)
    pitch_cp.set_as_design_variable(scaler=pitch_scaler)
    roll_cp.set_as_design_variable(scaler=roll_scaler)

    # ------------------------------------------------------------------
    # Evaluate splines
    # ------------------------------------------------------------------

    yaw_inputs = csdl.VariableGroup()
    yaw_inputs.yaw_cp = yaw_cp

    yaw = (
        BsplineComp(
            num_cp=num_cp,
            num_pt=num_times,
            jac=jac,
            in_name="yaw_cp",
            out_name="yaw",
        )
        .evaluate(yaw_inputs)
        .yaw
    )

    pitch_inputs = csdl.VariableGroup()
    pitch_inputs.pitch_cp = pitch_cp

    pitch = (
        BsplineComp(
            num_cp=num_cp,
            num_pt=num_times,
            jac=jac,
            in_name="pitch_cp",
            out_name="pitch",
        )
        .evaluate(pitch_inputs)
        .pitch
    )

    roll_inputs = csdl.VariableGroup()
    roll_inputs.roll_cp = roll_cp

    roll = (
        BsplineComp(
            num_cp=num_cp,
            num_pt=num_times,
            jac=jac,
            in_name="roll_cp",
            out_name="roll",
        )
        .evaluate(roll_inputs)
        .roll
    )

    # ------------------------------------------------------------------
    # Attitude boundary constraints
    # ------------------------------------------------------------------

    yaw[0].set_as_constraint(
        equals=initial_yaw,
        scaler=yaw_scaler,
    )

    pitch[0].set_as_constraint(
        equals=initial_pitch,
        scaler=pitch_scaler,
    )

    roll[0].set_as_constraint(
        equals=initial_roll,
        scaler=roll_scaler,
    )

    yaw[-1].set_as_constraint(
        equals=0.0,
        scaler=yaw_scaler,
    )

    pitch[-1].set_as_constraint(
        equals=0.0,
        scaler=pitch_scaler,
    )

    roll[-1].set_as_constraint(
        equals=0.0,
        scaler=roll_scaler,
    )

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    B_from_ECI = body123_reference_frame_change(
        yaw,
        pitch,
        roll,
        num_times,
    )

    B_from_RTN, B_from_ECI_dot = orbit_body_reference_frame_change(
        RTN_from_ECI,
        B_from_ECI,
        num_times,
        step_size,
    )

    rates, body_torque = body_rates(
        B_from_ECI,
        B_from_ECI_dot,
        osculating_orbit_angular_speed,
        sc_mmoi,
        step_size,
        num_times,
        gravity_gradient,
        B_from_RTN,
    )

    # ------------------------------------------------------------------
    # Fixed initial wheel speed
    # ------------------------------------------------------------------

    initial_rw_velocity = csdl.Variable(
        value=np.zeros(3),
        name="initial_reaction_wheel_velocity",
    )

    # NOT a design variable

    # ------------------------------------------------------------------
    # Integrate wheel dynamics
    # ------------------------------------------------------------------

    rw_velocity_history = runge_kutta_4(
        reaction_wheel_dynamics,
        initial_rw_velocity,
        rates,
        body_torque,
        step_size,
        num_times - 1,
        rw_mmoi,
    )

    # ------------------------------------------------------------------
    # Terminal wheel momentum constraint
    # ------------------------------------------------------------------

    rw_velocity_history[-1, :].set_as_constraint(
        equals=0.0,
        scaler=speed_scaler,
    )

    # ------------------------------------------------------------------
    # Wheel speed limits
    # ------------------------------------------------------------------

    rw_min_speed = csdl.minimum(rw_velocity_history)
    rw_max_speed = csdl.maximum(rw_velocity_history)

    rw_min_speed.set_as_constraint(
        lower=-max_rw_speed,
        scaler=speed_scaler,
    )

    rw_max_speed.set_as_constraint(
        upper=max_rw_speed,
        scaler=speed_scaler,
    )

    # ------------------------------------------------------------------
    # Wheel acceleration
    # ------------------------------------------------------------------

    rw_accel_history = csdl.Variable(
        value=np.zeros((num_times, 3))
    )

    rw_accel_history = rw_accel_history.set(
        csdl.slice[1:, :],
        (
            rw_velocity_history[1:, :]
            - rw_velocity_history[:-1, :]
        )
        / step_size,
    )

    # ------------------------------------------------------------------
    # Wheel torques
    # ------------------------------------------------------------------

    reaction_wheel_torque = csdl.Variable(
        value=np.zeros((num_times, 3))
    )

    reaction_wheel_torque = reaction_wheel_torque.set(
        csdl.slice[:, 0],
        rw_mmoi[0] * rw_accel_history[:, 0],
    )

    reaction_wheel_torque = reaction_wheel_torque.set(
        csdl.slice[:, 1],
        rw_mmoi[1] * rw_accel_history[:, 1],
    )

    reaction_wheel_torque = reaction_wheel_torque.set(
        csdl.slice[:, 2],
        rw_mmoi[2] * rw_accel_history[:, 2],
    )

    # ------------------------------------------------------------------
    # Torque limits
    # ------------------------------------------------------------------

    min_rw_torque = csdl.minimum(
        reaction_wheel_torque
    )

    max_rw_torque_used = csdl.maximum(
        reaction_wheel_torque
    )

    min_rw_torque.set_as_constraint(
        lower=-max_rw_torque,
        scaler=torque_scaler,
    )

    max_rw_torque_used.set_as_constraint(
        upper=max_rw_torque,
        scaler=torque_scaler,
    )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    rw_effort = (
        step_size
        * csdl.sum(
            reaction_wheel_torque ** 2
        )
    )

    # spline curvature penalty
    yaw_smooth = csdl.sum(
        (
            yaw[2:]
            - 2 * yaw[1:-1]
            + yaw[:-2]
        )
        ** 2
    )

    pitch_smooth = csdl.sum(
        (
            pitch[2:]
            - 2 * pitch[1:-1]
            + pitch[:-2]
        )
        ** 2
    )

    roll_smooth = csdl.sum(
        (
            roll[2:]
            - 2 * roll[1:-1]
            + roll[:-2]
        )
        ** 2
    )

    smoothness = 1e-3 * (
        yaw_smooth
        + pitch_smooth
        + roll_smooth
    )

    objective = rw_effort + smoothness

    objective.set_as_objective(
        scaler=1.0
    )

    return (
        rw_velocity_history,
        reaction_wheel_torque,
        yaw,
        pitch,
        roll,
    )




 
 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(0)
 
    num_times = 301
    num_cp = int((num_times - 1) / 5)
    duration = 95.
    step_size = duration * 60 / (num_times - 1)
 
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    RTN_from_ECI = csdl.Variable(
        value=np.tile(np.eye(3)[:, :, np.newaxis], (1, 1, num_times)),
        name='RTN_from_ECI'
    )
    osculating_orbit_angular_speed = csdl.Variable(
        value=np.ones((1, num_times)) * 0.001,
        name='osculating_orbit_angular_speed'
    )
 
    rw_vel, rw_torque, yaw, pitch, roll = attitude(
        num_times=num_times,
        num_cp=num_cp,
        step_size=step_size,
        RTN_from_ECI=RTN_from_ECI,
        osculating_orbit_angular_speed=osculating_orbit_angular_speed,
        gravity_gradient=True,
    )
 
    recorder.stop()
 
    # sim = csdl.experimental.JaxSimulator(recorder=recorder)
    # sim.check_totals()
    sim = csdl.experimental.PySimulator(recorder)
    sim.run()
 
    from modopt import CSDLAlphaProblem, SLSQP
    prob = CSDLAlphaProblem(problem_name='attitude_opt', simulator=sim)
    optimizer = SLSQP(prob, solver_options={'ftol': 1e-7, 'maxiter': 100})
 
 
    #print("Initial design variables:", prob.x0)
   # print("Initial objective:", prob.f_s)
    #print("Initial constraints:", prob.c_s)
 
    optimizer.solve()
    optimizer.print_results()
 
    print("Reaction wheel velocity history shape:", rw_vel.value.shape)
    print("First RW velocity:", rw_vel.value[0, :])
    print("Last RW velocity:", rw_vel.value[-1, :])
 
    # Time axis in minutes
    t_hist = np.arange(num_times) * step_size / 60
    labels = ['x', 'y', 'z']
 
    # Plot reaction wheel velocity
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for i in range(3):
        ax[i].plot(t_hist, rw_vel.value[:, i])
        ax[i].set_ylabel(f'RW velocity {labels[i]} (rad/s)')
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (minutes)')
    ax[0].set_title('Reaction Wheel Velocity History')
    plt.tight_layout()
    plt.show()
 
    # Plot reaction wheel torque
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for i in range(3):
        ax[i].plot(t_hist, rw_torque.value[:, i])
        ax[i].set_ylabel(f'RW torque {labels[i]} (Nm)')
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (minutes)')
    ax[0].set_title('Reaction Wheel Torque History')
    plt.tight_layout()
    plt.show()
 
    # Plot yaw, pitch, roll
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for i, (angle, name) in enumerate(zip([yaw, pitch, roll], ['Yaw', 'Pitch', 'Roll'])):
        ax[i].plot(t_hist, np.degrees(angle.value))
        ax[i].set_ylabel(f'{name} (degrees)')
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (minutes)')
    ax[0].set_title('Attitude Angles')
    plt.tight_layout()
    plt.show()