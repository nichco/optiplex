import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from modopt import JaxProblem, SLSQP
from beam_jax import Beam, CSTube
from lifting_line_jax import LiftingLine
import matplotlib.pyplot as plt
from optiplex import Plex
import modopt as mo
import warnings
warnings.filterwarnings("ignore")
import numpy as np


# aero setup
N = 31
b = 10.0
c_root = 1.0
c_tip = 0.65
lifting_line = LiftingLine(N, b, c_root, c_tip)
v_inf = 60
rho_atm = 1.225
q = 0.5 * rho_atm * v_inf**2

# structures setup
num_nodes   = 31
num_elems   = num_nodes - 1
mesh        = np.zeros((num_nodes, 3))
mesh[:, 1]  = lifting_line.y
fixed_nodes = [num_nodes // 2]
r = 0.2 * (lifting_line.chord[:-1] + lifting_line.chord[1:]) / 4
E = 69e9
G = 26e9
rho_mat = 3000
m0 = 1e3
load_factor = 5
safety_factor = 3
tip_disp_target = 0.01




def aero_subproblem(x, y, mu, args):

    twist = x[0]
    thickness = x[1]
    slack = x[2]
    beam_mass = args[0]

    def jax_obj(v):
        return obj + y.T @ c + 0.5 * mu * jnp.sum(c**2)

    return

def struct_subproblem(x, y, mu, args):

    twist = x[0]
    thickness = x[1]
    slack = x[2]

    def jax_obj(v):
        return obj + y.T @ c + 0.5 * mu * jnp.sum(c**2)

    return

def slack_update(x, y, mu):

    return

def con(v_init):

    return


opt = Plex(subproblems=[aero_subproblem, struct_subproblem, slack_update],
           x_init=x_init,
           con=con,
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=10,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-3, 
          RTOL_in=1e-3,
          ATOL_feas=1e-5,
          rho=1.1,
          mu=1.0,
          )