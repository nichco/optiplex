import csdl_alpha as csdl
import numpy as np
# from talos.utils.bspline_comp import BsplineComp, get_bspline_mtx
# from talos.disciplines.reference_frames.body123 import body123_reference_frame_change
# from talos.disciplines.attitude.orbit_body_reference_frame import orbit_body_reference_frame_change
import scipy.sparse
 
 
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
 
 
# class BsplineComp(csdl.CustomExplicitOperation):
#     """
#     Translates control points to actual points using a B-spline.
#     """
#     def __init__(self, num_cp, num_pt, jac, in_name, out_name):
#         super().__init__()
#         self.num_cp = num_cp
#         self.num_pt = num_pt
#         self.jac = jac
#         self.in_name = in_name
#         self.out_name = out_name
 
#     def evaluate(self, inputs: csdl.VariableGroup):
#         self.declare_input(self.in_name, getattr(inputs, self.in_name))
#         output = self.create_output(self.out_name, shape=(self.num_pt,))
#         self.declare_derivative_parameters(self.out_name, self.in_name, dependent=True)
#         out = csdl.VariableGroup()
#         setattr(out, self.out_name, output)
#         return out
 
#     def compute(self, input_vals, output_vals):
#         output_vals[self.out_name] = self.jac @ input_vals[self.in_name]
 
#     def compute_derivatives(self, input_vals, output_vals, derivatives):
#         derivatives[self.out_name, self.in_name] = self.jac.toarray().astype(float)
class BsplineComp(csdl.CustomExplicitOperation):
    """
    Translates control points to actual points using a B-spline.
    """
    def __init__(self, num_cp, num_pt, jac):
        super().__init__()
        self.num_cp = num_cp
        self.num_pt = num_pt
        self.jac = jac
 
    def evaluate(self, ycp):

        self.declare_input('ycp', ycp)

        yi = self.create_output('yi', shape=(self.num_pt,))

        return yi
 
    def compute(self, input_vals, output_vals):
        output_vals['yi'] = self.jac @ input_vals['ycp']
 
    def compute_derivatives(self, input_vals, output_vals, derivatives):
        derivatives['yi', 'ycp'] = self.jac.toarray().astype(float)
 
 
 
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



def attitude(num_times, num_cp, step_size, RTN_from_ECI, osculating_orbit_angular_speed,
             sc_mmoi=6 * np.array([2, 1, 3]) * 1e-3,
             rw_mmoi=6 * np.ones(3) * 1e-5,
             gravity_gradient=True):
 
    # if sc_mmoi.shape != (3,):
    #     raise ValueError('sc_mmoi must have shape (3,); has shape {}'.format(sc_mmoi.shape))
    # if rw_mmoi.shape != (3,):
    #     raise ValueError('rw_mmoi must have shape (3,); has shape {}'.format(rw_mmoi.shape))
 
    # B-spline control points
    jac = get_bspline_mtx(num_cp, num_times)
    yaw_cp = np.linspace(0, 1, num_cp)
    yaw_cp = csdl.Variable(value=yaw_cp)
    pitch_cp = csdl.Variable(value=np.linspace(0, 0, num_cp))
    roll_cp = csdl.Variable(value=np.linspace(0, 0, num_cp))
    # yaw_cp = csdl.Variable(value=np.zeros(num_cp), name='yaw_cp')
    # pitch_cp = csdl.Variable(value=np.zeros(num_cp), name='pitch_cp')
    # roll_cp = csdl.Variable(value=np.zeros(num_cp), name='roll_cp')

    yaw_cp.set_as_design_variable(lower=-np.deg2rad(15), upper=np.deg2rad(15))
    pitch_cp.set_as_design_variable(lower=-np.deg2rad(15), upper=np.deg2rad(15))
    roll_cp.set_as_design_variable(lower=-np.deg2rad(15), upper=np.deg2rad(15))
 
    yaw = BsplineComp(num_cp=num_cp, num_pt=num_times, jac=jac).evaluate(yaw_cp)
    
    initial_yaw = yaw[0]
    initial_yaw.set_as_constraint(equals=np.deg2rad(0), scaler=1e1)
    final_yaw = yaw[-1]
    final_yaw.set_as_constraint(equals=np.deg2rad(10), scaler=1e2)
 
    pitch = BsplineComp(num_cp=num_cp, num_pt=num_times, jac=jac).evaluate(pitch_cp)
    
    initial_pitch = pitch[0]
    initial_pitch.set_as_constraint(equals=np.deg2rad(0), scaler=1e1)
    final_pitch = pitch[-1]
    final_pitch.set_as_constraint(equals=np.deg2rad(10), scaler=1e2)
 
    roll = BsplineComp(num_cp=num_cp, num_pt=num_times, jac=jac).evaluate(roll_cp)
    
    initial_roll = roll[0]

    # Reference frame transformations
    B_from_ECI = body123_reference_frame_change(yaw, pitch, roll, num_times)
    B_from_RTN, B_from_ECI_dot = orbit_body_reference_frame_change(RTN_from_ECI, B_from_ECI, num_times, step_size)
    rates, body_torque = body_rates(B_from_ECI, B_from_ECI_dot, osculating_orbit_angular_speed,
                                    sc_mmoi, step_size, num_times, gravity_gradient, B_from_RTN)
 
    initial_rw_velocity = csdl.Variable(value=np.zeros(3), name='initial_reaction_wheel_velocity')
 
    # RK4 integration of reaction wheel dynamics
    rw_velocity_history = runge_kutta_4(
        reaction_wheel_dynamics,
        initial_rw_velocity,
        rates,
        body_torque,
        step_size,
        num_times - 1,
        rw_mmoi
    )
 
    # Reaction wheel acceleration and torque
    rw_accel_history = csdl.Variable(value=np.zeros((num_times, 3)))
    rw_accel_history = rw_accel_history.set(
        csdl.slice[1:, :],
        (rw_velocity_history[1:, :] - rw_velocity_history[:-1, :]) / step_size
    )
    reaction_wheel_torque = csdl.Variable(value=np.zeros((num_times, 3)))
    reaction_wheel_torque = reaction_wheel_torque.set(csdl.slice[:, 0], rw_mmoi[0] * rw_accel_history[:, 0])
    reaction_wheel_torque = reaction_wheel_torque.set(csdl.slice[:, 1], rw_mmoi[1] * rw_accel_history[:, 1])
    reaction_wheel_torque = reaction_wheel_torque.set(csdl.slice[:, 2], rw_mmoi[2] * rw_accel_history[:, 2])

    print("rw_velocity_history shape:", rw_velocity_history.shape)
    print("reaction_wheel_torque shape:", reaction_wheel_torque.shape)

    obj = csdl.sum(rw_velocity_history**2)
    obj.set_as_objective(scaler=1e3)
 
    return rw_velocity_history, reaction_wheel_torque, yaw, pitch, roll, yaw_cp, pitch_cp, roll_cp





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from modopt import CSDLAlphaProblem, SLSQP
 
    num_times = 301
    num_cp = int((num_times - 1) / 30)
    duration = 10.
    step_size = duration * 60 / (num_times - 1)
 
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    RTN_from_ECI = csdl.Variable(value=np.tile(np.eye(3)[:, :, np.newaxis], (1, 1, num_times)),)
    osculating_orbit_angular_speed = csdl.Variable(value=np.ones((1, num_times)) * 0.001)
    # osculating_orbit_angular_speed = csdl.Variable(value=np.ones((1, num_times)) * 0)
 
    rw_vel, rw_torque, yaw, pitch, roll, yaw_cp, pitch_cp, roll_cp = attitude(
        num_times=num_times,
        num_cp=num_cp,
        step_size=step_size,
        RTN_from_ECI=RTN_from_ECI,
        osculating_orbit_angular_speed=osculating_orbit_angular_speed,
        gravity_gradient=True,
    )
 
    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder=recorder)
    prob = CSDLAlphaProblem(simulator=sim)
    optimizer = SLSQP(prob, solver_options={'ftol': 1e-7, 'maxiter': 100}, turn_off_outputs=True)
 
    optimizer.solve()
    optimizer.print_results()


    recorder.execute()
    # print(yaw.value)
 
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
 
    # Plot yaw, pitch, roll and the control points
    yaw_cp = yaw_cp.value
    pitch_cp = pitch_cp.value
    roll_cp = roll_cp.value
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for i, (angle, name) in enumerate(zip([yaw, pitch, roll], ['Yaw', 'Pitch', 'Roll'])):
        ax[i].plot(t_hist, np.degrees(angle.value))
        ax[i].set_ylabel(f'{name} (degrees)')
        ax[i].grid(True)
        ax[i].scatter(np.linspace(0, duration, num_cp), np.degrees([yaw_cp, pitch_cp, roll_cp][i]), color='red', label='Control Points')
    ax[-1].set_xlabel('Time (minutes)')
    ax[0].set_title('Attitude Angles')
    plt.tight_layout()
    plt.show()