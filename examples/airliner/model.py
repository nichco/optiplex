import jax.numpy as jnp
import jax
import numpy as np
from interpax_atm1976 import akima_interp_a, akima_interp_rho, akima_interp_T




softplus = lambda x, k=30: jnp.log1p(jnp.exp(k * x)) / k

def wing_mass_model(AR, S, sweep=np.deg2rad(35), tc=0.15, n_ult=1.5 * 4, K=0.003, rho_mat=2700):
    # Sadraey or Torenbeek
    MAC = jnp.sqrt(S / AR)
    return S**0.6 * MAC * tc * rho_mat * K * (AR * n_ult / jnp.cos(sweep))**0.6 * 9.81



def max_thrust_model(rho, tmax_sl, mach, rho_sl=1.225):
    tmax = tmax_sl * (rho / rho_sl) # altitude variation

    tmax *= (1 - 0.5 * mach + 0.2 * mach**2) # concave-up quadratic decrease

    return jnp.max(jnp.array([0, tmax]))


def tsfc_model(mach, 
               temp, 
               eta,
               temp_sl=288.15,
               ):

    tsfc = (temp / temp_sl)**0.5 * (0.4 + 0.45 * mach) # lb fuel / (hr * lbf thrust)

    # tsfc *= 1 - 0.4*eta # throttle model induces bang-bang control
    tsfc *= 0.7 + 0.5*(1.3 - eta)**4 # no longer bang-bang, more like whoosh-whoosh

    return tsfc * (1/3600) * (1/4.44822162) * 0.45359237 # kg fuel / (s * N thrust)


def korn(CL, tc=0.15, kappa=0.8, sweep=np.deg2rad(35)):
    csweep = np.cos(sweep)
    tech = kappa / csweep
    thick = tc / csweep**2
    lift = CL / (10 * csweep**3)
    MDD = tech - thick - lift
    Mcrit = MDD - (0.1 / 80)**(1/3)
    return Mcrit


# dynamics function
def f(t, y, args):

    h, r, v, gamma, m = y

    eta, theta, tf, AR, S, nu = args

    eta = jnp.interp(t, jnp.linspace(0, tf, nu), eta)
    theta = jnp.interp(t, jnp.linspace(0, tf, nu), theta)

    rho = akima_interp_rho(h) # kg/m^3
    sos = akima_interp_a(h) # m/s
    ambient_temperature = akima_interp_T(h) # K

    mach = v / sos
    q = 0.5 * rho * v**2

    # beta = (softplus(1 - mach**2, k=10))**0.5
    CL_alpha = 0.8 * 2 * np.pi #/ beta
    alpha = theta - gamma
    CL = CL_alpha * alpha

    CLmax = 1.2 # saturate CL at CLmax
    CL = CLmax * jnp.tanh(CL / CLmax)

    L = q * S * CL

    CD0 = 0.02
    e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64
    CD = CD0 + (CL**2 / (np.pi * e * AR))

    Mcrit = korn(CL)
    CD += 20 * softplus(mach - Mcrit, k=50)**4 # wave drag

    D = CD * q * S

    max_thrust = max_thrust_model(rho, tmax_sl=100000, mach=mach) # N
    T = eta * max_thrust

    tsfc = tsfc_model(mach, ambient_temperature, eta)

    # equations of motion
    h_dot = v * jnp.sin(gamma)
    r_dot = v * jnp.cos(gamma)
    v_dot = (1 / m) * (T * jnp.cos(alpha) - D) - 9.81 * jnp.sin(gamma)
    gamma_dot = (1 / v) * (((T * jnp.sin(alpha) + L) / m) - 9.81 * jnp.cos(gamma))
    m_dot = -1 * tsfc * T

    return jnp.array([h_dot, r_dot, v_dot, gamma_dot, m_dot])








def compute_objective(AR, S, eta, theta, tf, fuel, data):

    nu = data.nu
    v0 = data.v0
    nt = data.nt
    payload = 25000

    empty = wing_mass_model(AR, S) # compute m0
    m0 = payload + fuel + empty # (kg)

    args = [eta, theta, tf, AR, S, nu]
    y0 = jnp.array([3048.0, 0.0, v0, 0.0, m0])
    sol, _ = jax_rk4(f, t0=0, y0=y0, h=tf / nt, n=nt, args=args)

    m = sol[:, 4].ravel()
    fuel_used = m0 - m[-1]

    return fuel_used # minimize fuel burn




def compute_constraints(AR, S, eta, theta, tf, fuel, data):

    nu = data.nu
    v0 = data.v0
    nt = data.nt
    payload = 25000

    empty = wing_mass_model(AR, S) # compute m0
    m0 = payload + fuel + empty # (kg)

    args = [eta, theta, tf, AR, S, nu]
    y0 = jnp.array([3048.0, 0.0, v0, 0.0, m0])
    sol, rhs = jax_rk4(f, t0=0, y0=y0, h=tf / nt, n=nt, args=args)

    h = sol[:, 0].ravel()
    r = sol[:, 1].ravel()
    m = sol[:, 4].ravel()

    mf_constraint = (m0 - m[-1]) - fuel

    con = jnp.zeros((3))
    con = con.at[0].set(h[-1])
    con = con.at[1].set(r[-1])
    con = con.at[2].set(mf_constraint)

    return con












def jax_rk4(f, t0, y0, h, n, args):

    y0 = jnp.atleast_1d(y0)
    m = len(y0)

    solution = jnp.zeros((n + 1, m))
    rhs_vals = jnp.zeros((n + 1, m))

    # initial condition
    solution = solution.at[0].set(y0)
    rhs_vals = rhs_vals.at[0].set(f(t0, y0, args))

    def step(carry, _):
        t, Y = carry

        f0 = f(t, Y, args)        # RHS at current step
        k1 = h * f0

        t_half = t + 0.5 * h
        t_next = t + h

        k2 = h * f(t_half, Y + 0.5 * k1, args)
        k3 = h * f(t_half, Y + 0.5 * k2, args)
        k4 = h * f(t_next, Y + k3, args)

        Y_next = Y + (k1 + 2*k2 + 2*k3 + k4) / 6
        f_next = f(t_next, Y_next, args)

        return (t_next, Y_next), (Y_next, f_next)

    # run integrator
    (_, _), (ys, fs) = jax.lax.scan(step, (t0, y0), None, length=n)

    solution = solution.at[1:].set(ys)
    rhs_vals = rhs_vals.at[1:].set(fs)

    return solution, rhs_vals