import numpy as np
import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)
import gc
from optiplex import combo
from model import compute_objective, compute_constraints
from modopt import JaxProblem, SLSQP, IPOPT
from meta_data import Params


def make_sub_problem(i, r, N):

    def subP_i(v_init: list,
               y: np.ndarray = None, # lagrange multipliers
               mu: float = 1, # penalty coefficient
               ) -> list:
        
        # v_init = [AR_1, ..., AR_N, S_1, ..., S_N, d_1, ..., d_N]

        nu = Params[r].nu

        AR_list = [v_init[i] for i in range(N)] # the first N values in v_init are AR global vars
        S_list = [v_init[i + N] for i in range(N)] # the next N values in v_init are S global vars
        d_list = v_init[2*N:] # the last values are local variable vectors

        def jax_obj(v):
            # order of vars: eta_i, theta_i, tf_i, AR_i, S_i, fuel_i
            eta_i = v[:nu]
            theta_i = v[nu:-4]
            tf_i = v[-4]
            AR_i = v[-3]
            S_i = v[-2]
            fuel_i = v[-1]

            d_i = jnp.concatenate((eta_i, theta_i, tf_i, fuel_i))
            # d_i order: eta, theta, tf, fuel

            AR_list[i] = AR_i
            S_list[i] = S_i

            obj = compute_objective(AR_i, S_i, d_i, Params[r])

            AR_constraint, S_constraint = combo(AR_list), combo(S_list)
            c = jnp.concatenate((AR_constraint, S_constraint))
            return obj + y.T @ c + mu * jnp.sum(c**2)
        
        def jax_con(v):
            # order of vars: eta_i, theta_i, tf_i, AR_i, S_i, fuel_i
            eta_i = v[:nu]
            theta_i = v[nu:-4]
            tf_i = v[-4]
            AR_i = v[-3]
            S_i = v[-2]
            fuel_i = v[-1]

            d_i = jnp.concatenate((eta_i, theta_i, tf_i, fuel_i))
            # d_i order: eta, theta, tf, fuel
            
            return compute_constraints(AR_i, S_i, d_i, Params[r])
        


        x0 = np.concatenate((AR_list[i], S_list[i], d_list[i]))

        eta_scale, theta_scale, tf_scale, ar_scale, s_scale, f_scale = 1, 1e1, 1e-3, 1, 1e-1, 1e-3

        nu = Params[r].nu
        x_scaler = np.concatenate((np.full((nu), eta_scale), np.full((nu), theta_scale), np.array([tf_scale]), np.array([ar_scale]), np.array([s_scale]), np.array([f_scale])))
        c_scaler = np.array([1e-3, 1e-6, 1e-3])
        cl = cu = np.array([3048, r, 0.])

        # variable bounds
        eta_l, eta_u = np.full((nu), 0.0), np.full((nu), 1.0)
        theta_l, theta_u = np.full((nu), np.deg2rad(-5)), np.full((nu), np.deg2rad(15))
        tf_l, tf_u = np.array([1e3]), np.array([1e5])
        AR_l, AR_u = np.array([5.0]), np.array([50.0])
        S_l, S_u = np.array([30.0]), np.array([150.0])
        fuel_l, fuel_u = np.array([300.0]), np.array([10000.0])
        xl, xu = np.concatenate((eta_l, theta_l, tf_l, AR_l, S_l, fuel_l)), np.concatenate((eta_u, theta_u, tf_u, AR_u, S_u, fuel_u))

        jaxprob = JaxProblem(x0=x0, jax_obj=jax_obj, jax_con=jax_con, xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=1e-4)
        optimizer = IPOPT(jaxprob, solver_options={'max_iter': 300, 'tol': 1e-7}, turn_off_outputs=True)
        optimizer.solve()
        optimizer.print_results()
        ans = optimizer.results['x']

        # update lists with subPi results
        # order of vars: eta_i, theta_i, tf_i, AR_i, S_i, fuel_i
            # eta_i = v[:nu]
            # theta_i = v[nu:-4]
            # tf_i = v[-4]
            # AR_i = v[-3]
            # S_i = v[-2]
            # fuel_i = v[-1]
        
        # AR_list[i] = ans[0]
        # S_list[i] = ans[1]
        # d_list[i] = ans[2:]

        d_i = jnp.concatenate((eta_i, theta_i, tf_i, fuel_i))

        AR_list[i] = ans[0]
        S_list[i] = ans[1]
        d_list[i] = ans[2:]

        gc.collect()
        return AR_list + S_list + d_list



    return subP_i