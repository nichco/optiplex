import numpy as np
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
import modopt as mo
import gc
from optiplex import combo
from model import compute_objective, compute_constraints
from modopt import JaxProblem, SLSQP, IPOPT
from meta_data import Params


def make_sub_problem(i, r):

    def subP_i(v_init: list,
                      y: np.ndarray = None, # lagrange multipliers
                      mu: float = 1, # penalty coefficient
                      ) -> list:


        def jax_obj(v):
            AR_i = v[0]
            S_i = v[1]
            d_i = v[2:]
            AR_list[i] = AR_i
            S_list[i] = S_i

            obj = compute_objective(...)

            AR_constraint, S_constraint = combo(AR_list), combo(S_list)
            c = jnp.concatenate((AR_constraint, S_constraint))
            return obj + y.T @ c + mu * jnp.sum(c**2)
        
        def jax_con(v):
            
            return compute_constraints(...)
        

        eta_scale = 1
        theta_scale = 1e1
        tf_scale = 1e-3
        ar_scale = 1
        s_scale = 1e-1
        f_scale = 1e-3

        x_scaler = np.concatenate((np.full((nu), eta_scale),
                                np.full((nu), theta_scale),
                                np.array([tf_scale]),
                                np.array([ar_scale]),
                                np.array([s_scale]),
                                np.array([f_scale]),
                                ))

        c_scaler = np.array([1e-3, 1e-6, 1e-3])
        cl = cu = np.array([3048, r, 0.])

        jaxprob = JaxProblem(x0=x0, jax_obj=jax_obj, jax_con=jax_con, xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=1e-4)
        optimizer = IPOPT(jaxprob, solver_options={'max_iter': 300, 'tol': 1e-7}, turn_off_outputs=True)
        optimizer.solve()
        optimizer.print_results()

        gc.collect()
        return l_list + mp_list + x_list + u_list



    return subP_i