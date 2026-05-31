from lifting_line_jax_3 import LiftingLine
from beam_jax import Beam, CSTube
import numpy as np
from crm_mesh import build_crm_mesh
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv
import jax
jax.config.update("jax_enable_x64", True)
from modopt import JaxProblem, SLSQP
from scipy.stats.qmc import LatinHypercube, scale
import gc


def make_subproblem(subP, rho_atm_i, v_inf_i, num):

    def subP_i(v_init: list,
               y: np.ndarray = None, # lagrange multipliers
               mu: float = 1, # penalty parameter
               ) -> list:
        
        alphas = v_init[0] # trim angles for all conditions
        twists = v_init[1] # twist distributions for all conditions
        thicknesses = v_init[2] # thickness distributions for all conditions


        def objective(x):
            alpha_i = x[0] # trim angle for this condition
            twist_i = x[1:1 + ns] # twist distribution for this condition
            thickness_i = x[1 + ns:] # thickness distribution for this condition

            alphas[subP] = alpha_i
            twists[subP] = twist_i
            thicknesses[subP] = thickness_i

            effective_twist = twist_i + alpha_i

            ll = LiftingLine(le, te, v_inf_i, rho_atm_i)
            sol = ll.solve_lifting_line_model(effective_twist)
            CD = sol["CD"]

            L = 1e2 * CD + y.T @ c + 0.5 * mu * jnp.sum(c**2)
            return L


        def constraints(x):

            alpha_i = x[0] # trim angle for this condition
            twist_i = x[1:1 + ns] # twist distribution for this condition
            thickness_i = x[1 + ns:] # thickness distribution for this condition

            alphas[subP] = alpha_i
            twists[subP] = twist_i
            thicknesses[subP] = thickness_i

            _, con_i = condition(rho_atm, v_inf, alpha_i, twist, thickness)

            return con_i

        gc.collect()
        return []



    return subP_i