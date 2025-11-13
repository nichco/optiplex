import jax.numpy as jnp
import jax
import numpy as np
from modopt import JaxProblem, SLSQP, IPOPT
jax.config.update("jax_enable_x64", True)
from interpax_aero_models import _interp_CL_alpha, _interp_CD_0, _interp_K
from interpax_atm1976 import akima_interp_a, akima_interp_rho
# from b777_engine.interpax_b777_engine import interp_tmax, interp_sfc
from b777_engine.interpax_b777_engine import interp_sfc

g = 9.81
nu = 60

def f(t, y, args):

    h, r, v, gamma, m = y

    eta, theta, tf, S = args
    eta = jnp.interp(t, jnp.linspace(0, tf, nu), eta)
    theta = jnp.interp(t, jnp.linspace(0, tf, nu), theta)

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
    # sfc = interp_sfc(clipped_mach, h_km_clipped, eta) # kg/N/s
    sfc = interp_sfc(mach, h_km_clipped, eta) # kg/N/s

    # equations of motion
    h_dot = v * jnp.sin(gamma)
    r_dot = v * jnp.cos(gamma)
    v_dot = (1 / m) * (T * jnp.cos(alpha) - D) - g * jnp.sin(gamma)
    gamma_dot = (1 / v) * (((T * jnp.sin(alpha) + L) / m) - g * jnp.cos(gamma))
    m_dot = -1 * sfc * T

    return jnp.array([h_dot, r_dot, v_dot, gamma_dot, m_dot])