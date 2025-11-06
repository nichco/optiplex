import jax.numpy as jnp
import jax
import numpy as np
from modopt import JaxProblem, SLSQP, IPOPT
jax.config.update("jax_enable_x64", True)
from interpax_aero_models import _interp_CL_alpha, _interp_CD_0, _interp_K
from interpax_atm1976 import akima_interp_a, akima_interp_rho
from vanilla_rk4 import jax_rk4
from b777_engine.interpax_b777_engine import interp_tmax, interp_sfc
from plot import plot_trajectory
from mass_fun import _mass
import pickle

t0 = 0.0
nu = 60 # control n
num_steps = 6000 # num steps
g = 9.81 # m/s^2

# variable scaling
eta_scale = 1.0
theta_scale = 1e1
tf_scale = 1e-3
b_scale = 1e-1

# y0 for ODE solver
h0 = 3048.0 # m (10,000 ft)
r0 = 0.0 # m
v0 = 147.6 # m/s
gamma0 = 0.0 # rad

# terminal conditions
hf = 3048.0 # m (10,000 ft)
rf = 2e6 # 2.5e6 # m (fixed range)
vf = 147.6 # m/s (final velocity)

payload_mass = 10000 # (kg)
not_wing_mass = 20000 # (kg)
# wing_span = 32 # (m)
mean_aerodynamic_chord = 3 # (m)


# dynamics function
def f(t, y, args):

    h, r, v, gamma, m = y

    eta, theta, tf, S = args
    eta = jnp.interp(t, jnp.linspace(t0, tf, nu), eta)
    theta = jnp.interp(t, jnp.linspace(t0, tf, nu), theta)

    rho = akima_interp_rho(h) # kg/m^3
    sos = akima_interp_a(h) # m/s
    mach = v / sos
    q = 0.5 * rho * v**2

    CL_alpha = _interp_CL_alpha(mach)
    CD_0 = _interp_CD_0(mach)
    K = _interp_K(mach)

    # ************************************************** CL_alpha adjusted as S changes (experimental) **************************************************
    CL_alpha = CL_alpha + 1e-1 * (S - 71.0)

    alpha = theta - gamma

    CL = CL_alpha * alpha
    L = q * S * CL

    CD = CD_0 + K * CL**2
    D = q * S * CD

    h_ft = h * 3.281
    max_thrust = ((1 - mach) * (1 - (h_ft / 4e4)) * 28.5 +
                  mach * (1 - (h_ft / 4e4)) * 29.3 +
                  (1 - mach) * (h_ft / 4e4) * 7.4 +
                  mach * (h_ft / 4e4) * 10.7) * 1e3  # lbf
    max_thrust *= 4.44822162 # N
    T = eta * max_thrust

    # T = interp_tmax(mach, h / 1e3, eta) # N
    # T = interp_tmax(jnp.clip(mach, 0.0, 0.9), jnp.clip(h / 1e3, 0.0, 15.0), eta) # N

    # sfc = interp_sfc(mach, h / 1e3, eta)
    h_km_clipped = jnp.clip(h / 1e3, 0.0, 15.0)
    # sfc = interp_sfc(jnp.clip(mach, 0.0, 1.0), h_km_clipped, eta) # kg/N/s
    sfc = interp_sfc(mach, h_km_clipped, eta) # kg/N/s

    # equations of motion
    h_dot = v * jnp.sin(gamma)
    r_dot = v * jnp.cos(gamma)
    v_dot = (1 / m) * (T * jnp.cos(alpha) - D) - g * jnp.sin(gamma)
    gamma_dot = (1 / v) * (((T * jnp.sin(alpha) + L) / m) - g * jnp.cos(gamma))
    m_dot = -1 * sfc * T

    return jnp.array([h_dot, r_dot, v_dot, gamma_dot, m_dot])



# objective function for ModOpt
def jax_obj(d):

    # scaled variables
    eta = d[:nu]
    theta = d[nu:-2]
    tf = d[-2]
    b = d[-1]

    # unscale variables
    eta = eta / eta_scale
    theta = theta / theta_scale
    tf = tf / tf_scale
    b = b / b_scale

    wing_area = b * mean_aerodynamic_chord
    m0 = _mass(payload_mass, not_wing_mass, wing_area, b)

    args = [eta, theta, tf, wing_area]

    h = tf / num_steps
    y0 = jnp.array([h0, r0, v0, gamma0, m0])
    sol = jax_rk4(f, t0=t0, y0=y0, h=h, n=num_steps, args=args)

    # solution
    # h = sol.ys[:, 0].ravel()
    # r = sol.ys[:, 1].ravel()
    # v = sol.ys[:, 2].ravel()
    # gamma = sol.ys[:, 3].ravel()
    # m = sol.ys[:, 4].ravel()
    m = sol[:, 4].ravel()
    m_f = m[-1]

    fuel_used = m0 - m_f

    return 1e-3 * fuel_used


def jax_con(d):

    # scaled variables
    eta = d[:nu]
    theta = d[nu:-2]
    tf = d[-2]
    b = d[-1]

    # unscale variables
    eta = eta / eta_scale
    theta = theta / theta_scale
    tf = tf / tf_scale
    b = b / b_scale

    wing_area = b * mean_aerodynamic_chord
    m0 = _mass(payload_mass, not_wing_mass, wing_area, b)

    args = [eta, theta, tf, wing_area]

    h = tf / num_steps
    y0 = jnp.array([h0, r0, v0, gamma0, m0])
    sol = jax_rk4(f, t0=t0, y0=y0, h=h, n=num_steps, args=args)

    # solution
    h = sol[:, 0].ravel()
    r = sol[:, 1].ravel()
    v = sol[:, 2].ravel()
    gamma = sol[:, 3].ravel()
    m = sol[:, 4].ravel()

    # terminal constraints
    hf_constraint = h[-1] - hf
    rf_constraint = r[-1] - rf
    vf_constraint = v[-1] - vf

    con = jnp.zeros((3))
    con = con.at[0].set(hf_constraint * 1e-3)
    con = con.at[1].set(rf_constraint * 1e-6)
    con = con.at[2].set(vf_constraint * 1e-2)

    return con


# variable bounds for ModOpt
eta_l = np.full((nu), 0.0 * eta_scale)
eta_u = np.full((nu), 1.0 * eta_scale)

theta_l = np.full((nu), np.deg2rad(-10) * theta_scale)
theta_u = np.full((nu), np.deg2rad(15) * theta_scale)

tf_l = np.array([1e3]) * tf_scale
tf_u = np.array([np.inf])

b_l = np.array([10.0]) * b_scale
b_u = np.array([50.0]) * b_scale

xl = np.concatenate((eta_l, theta_l, tf_l, b_l))
xu = np.concatenate((eta_u, theta_u, tf_u, b_u))


# initial guess for ModOpt
eta0 = np.linspace(0.6, 0.2, nu) * eta_scale
theta0 = np.linspace(np.deg2rad(3), np.deg2rad(1), nu) * theta_scale
tf0 = np.array([7000.0]) * tf_scale
b0 = np.array([32.0]) * b_scale
x0 = np.concatenate((eta0, theta0, tf0, b0))

# with open('data.pkl', 'rb') as file:
#     data = pickle.load(file)

# eta0 = data['eta'] * eta_scale
# theta0 = data['theta'] * theta_scale
# tf0 = data['tf'] * tf_scale
# b0 = data['b'] * b_scale

# tf0 = np.atleast_1d(tf0)
# b0 = np.atleast_1d(b0)

# x0 = np.concatenate((eta0, theta0, np.atleast_1d(tf0), np.atleast_1d(b0)))


jaxprob = JaxProblem(x0=x0, jax_obj=jax_obj, jax_con=jax_con, 
                     order=1, xl=xl, xu=xu, cl=0., cu=0.)


# optimizer = SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-6}, turn_off_outputs=True)
optimizer = IPOPT(jaxprob, solver_options={'max_iter': 300, 'tol': 1e-5}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()


x = optimizer.results['x']
eta = x[:nu] / eta_scale
theta = x[nu:-2] / theta_scale
tf = x[-2] / tf_scale
b = x[-1] / b_scale

print('tf: ', tf)
print('b: ', b)


# simulate with optimal control
wing_area = b * mean_aerodynamic_chord
m0 = _mass(payload_mass, not_wing_mass, wing_area, b)

args = [eta, theta, tf, wing_area]
h = tf / num_steps
y0 = jnp.array([h0, r0, v0, gamma0, m0])
sol = jax_rk4(f, t0=t0, y0=y0, h=h, n=num_steps, args=args)

# solution
h = sol[:, 0].ravel()
r = sol[:, 1].ravel()
v = sol[:, 2].ravel()
gamma = sol[:, 3].ravel()
m = sol[:, 4].ravel()


# import pickle
# with open('data.pkl', 'wb') as f:
#     pickle.dump({'h': h, 'r': r, 'v': v, 'gamma': gamma, 'm': m,
#                  'eta': eta, 'theta': theta, 'tf': tf, 'b': b}, f)


plot_trajectory(h, r, v, gamma, m, eta, theta, tf)


# 448.8 seconds (plugged in vanilla_rk4 jax_rk4)
# 429.4 seconds (battery ode_dev jax_rk4 with .set)
# 408.3 seconds (battery ode_dev jax_rk4 with return at.set and t_plus_half_h)
# 401.7 seconds (battery vanilla_rk4 with t_plus_h)
# 415.1 seconds (battery vanilla_rk4 with qS)
# 362.5 seconds (plugged in vanilla_rk4 jax_rk4 with t_plus_h)