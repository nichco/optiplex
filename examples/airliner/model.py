import jax.numpy as jnp
import jax
import numpy as np
from interpax_atm1976 import akima_interp_a, akima_interp_rho, akima_interp_T
import jaxopt



softplus = lambda x, k=30: jnp.log1p(jnp.exp(k * x)) / k

def wing_mass_model(AR, S, sweep=np.deg2rad(35), tc=0.15, n_ult=1.5 * 4, K=0.003, rho_mat=2700):
    MAC = jnp.sqrt(S / AR)
    return S**0.6 * MAC * tc * rho_mat * K * (AR * n_ult / jnp.cos(sweep))**0.6 * 9.81


def korn(CL, tc=0.15, kappa=0.8, sweep=np.deg2rad(35)):
    csweep = np.cos(sweep)
    tech = kappa / csweep
    thick = tc / csweep**2
    lift = CL / (10 * csweep**3)
    MDD = tech - thick - lift
    Mcrit = MDD - (0.1 / 80)**(1/3)
    return Mcrit


def tsfc_model(mach, 
               temp, 
               eta,
               temp_sl=288.15,
               ):

    tsfc = (temp / temp_sl)**0.5 * (0.4 + 0.45 * mach) # lb fuel / (hr * lbf thrust)

    tsfc *= 0.7 + 0.5*(1.3 - eta)**4 # no longer bang-bang, more like whoosh-whoosh

    return tsfc * 2.832545e-5 # kg fuel / (s * N thrust)


def korn(CL, tc=0.15, kappa=0.8, sweep=np.deg2rad(35)):
    csweep = np.cos(sweep)
    tech = kappa / csweep
    thick = tc / csweep**2
    lift = CL / (10 * csweep**3)
    MDD = tech - thick - lift
    return MDD - (0.1 / 80)**(1/3) # Mcrit


# dynamics function
def f(t, y, args):
    nu = 300

    h, r, v, gamma, m = y
    eta, theta, tf, AR, S = args

    xp = jnp.linspace(0, tf, nu)
    eta = jnp.interp(t, xp, eta)
    theta = jnp.interp(t, xp, theta)

    rho = akima_interp_rho(h) # kg/m^3
    sos = akima_interp_a(h) # m/s
    ambient_temperature = akima_interp_T(h) # K

    mach = v / sos
    q = 0.5 * rho * v**2

    CL_alpha = 0.8 * 2 * np.pi
    alpha = theta - gamma
    CL = CL_alpha * alpha

    CLmax = 1.2 # saturate CL at CLmax
    CL = CLmax * jnp.tanh(CL / CLmax)

    L = q * S * CL

    e = 1.78 * (1 - 0.045 * AR**0.68) - 0.64
    CD = 0.02 + (CL**2 / (np.pi * e * AR))
    Mcrit = korn(CL)
    CD += 20 * softplus(mach - Mcrit, k=50)**4 # wave drag

    D = CD * q * S

    tmax_sl = 100000 # N
    tmax = tmax_sl * (rho / 1.225)
    T = eta * tmax

    tsfc = tsfc_model(mach, ambient_temperature, eta)

    # equations of motion
    h_dot = v * jnp.sin(gamma)
    r_dot = v * jnp.cos(gamma)
    v_dot = (1 / m) * (T * jnp.cos(alpha) - D) - 9.81 * jnp.sin(gamma)
    gamma_dot = (1 / v) * (((T * jnp.sin(alpha) + L) / m) - 9.81 * jnp.cos(gamma))
    m_dot = -1 * tsfc * T

    return jnp.array([h_dot, r_dot, v_dot, gamma_dot, m_dot])








def compute_objective(AR, S, eta, theta, tf, fuel, data):

    nt = 500

    v0 = data.v0
    payload = 25000

    empty = wing_mass_model(AR, S) # compute m0
    m0 = payload + fuel + empty # (kg)

    args = [eta, theta, tf, AR, S]
    y0 = jnp.array([3048.0, 0.0, v0, 0.0, m0])
    sol = backward_euler_newton(f, t0=0, y0=y0, h=tf / nt, n=nt, args=args)

    m = sol[:, 4].ravel()
    fuel_used = m0 - m[-1]

    return fuel_used # minimize fuel burn




def compute_constraints(AR, S, eta, theta, tf, fuel, data):

    nt = 500

    v0 = data.v0
    payload = 25000

    empty = wing_mass_model(AR, S) # compute m0
    m0 = payload + fuel + empty # (kg)

    args = [eta, theta, tf, AR, S]
    y0 = jnp.array([3048.0, 0.0, v0, 0.0, m0])
    sol = backward_euler_newton(f, t0=0, y0=y0, h=tf / nt, n=nt, args=args)

    h = sol[:, 0].ravel()
    r = sol[:, 1].ravel()
    m = sol[:, 4].ravel()

    # mf_constraint = (m0 - m[-1]) - fuel
    mf_constraint = (m0 - m[-1]) / fuel

    con = jnp.zeros((3))
    con = con.at[0].set(h[-1])
    con = con.at[1].set(r[-1])
    con = con.at[2].set(mf_constraint)

    return con












def newton_fpi(G, y_init, data, tol=1e-9, maxiter=1000):

    def T(y, data): # Define the Newton fixed point map
        res = G(y, data) # Compute residual
        J = jax.jacrev(G)(y, data) # Compute Jacobian
        delta = jnp.linalg.solve(J, -res) # Newton step
        return y + delta

    fpi = jaxopt.FixedPointIteration(fixed_point_fun=T,
                                     maxiter=maxiter,
                                     tol=tol,
                                     )

    return fpi.run(y_init, data).params


def backward_euler_step(y_prev, t_next, h, args, f):

    def G(y_next, data): # Residual function with all differentiable values explicit
        y_prev, t_next, h, args = data
        return y_next - y_prev - h * f(t_next, y_next, args)

    y_init = y_prev

    data = (y_prev, t_next, h, args)
    y_next = newton_fpi(G, y_init, data)
    return y_next

def backward_euler_newton(f, t0, y0, h, n, args):

    y0 = jnp.atleast_1d(y0)

    def step(carry, _):
        t_prev, y_prev = carry
        t_next = t_prev + h
        y_next = backward_euler_step(y_prev, t_next, h, args, f)
        return (t_next, y_next), y_next

    (_, _), ys = jax.lax.scan(step, (t0, y0), None, length=n)

    return jnp.vstack([y0, ys])