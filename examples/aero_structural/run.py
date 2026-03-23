from optiplex import Plex
import numpy as np
import modopt as mo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def aero_subproblem(x, y, mu):

    return

def struct_subproblem(x, y, mu):

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