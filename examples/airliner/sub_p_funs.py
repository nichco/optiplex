import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import gc
from optiplex import combo
from model import compute_objective, compute_constraints
from modopt import JaxProblem, SLSQP
import warnings
warnings.filterwarnings("ignore")
import time


def make_sub_problem(subP, r, N, data):

    def subP_i(v_init: list,
               y: np.ndarray = None, # lagrange multipliers
               mu: float = 1, # penalty coefficient
               ) -> list:
        
        print('Solving subproblem ', subP, ' with range ', r)

        # nu = 300
        n_eta = 50
        n_theta = 50

        AR_list = [v_init[j][-3] for j in range(N)]
        S_list  = [v_init[j][-2] for j in range(N)]

        def jax_obj(v):
            # order of vars: eta_i, theta_i, tf_i, AR_i, S_i, fuel_i
            eta_i = v[:n_eta]
            theta_i = v[n_eta:-4]
            tf_i = v[-4]
            AR_i = v[-3]
            S_i = v[-2]
            fuel_i = v[-1]

            AR_list[subP] = AR_i
            S_list[subP] = S_i

            obj = compute_objective(AR_i, S_i, eta_i, theta_i, tf_i, fuel_i)

            AR_constraint, S_constraint = combo(AR_list), combo(S_list)
            c = jnp.concatenate((AR_constraint, S_constraint)) / 1e2

            return 1e-3 * obj + y.T @ c + 0.5 * mu * jnp.sum(c**2)
            # return 1e-3 * obj
        
        def jax_con(v):
            # order of vars: eta_i, theta_i, tf_i, AR_i, S_i, fuel_i
            eta_i = v[:n_eta]
            theta_i = v[n_eta:-4]
            tf_i = v[-4]
            AR_i = v[-3]
            S_i = v[-2]
            fuel_i = v[-1]
            
            return compute_constraints(AR_i, S_i, eta_i, theta_i, tf_i, fuel_i)
        

        x0 = v_init[subP]
        
        x_scaler = np.concatenate((np.full((n_eta), 1), # eta scale
                                   np.full((n_theta), 1e1), # theta scale
                                   np.array([1 / 20000]), # tf scale
                                   np.array([1]), # AR scale
                                   np.array([1e-1]), # S scale
                                   np.array([1 / 4000]) # fuel scale
                                   ))
        
        c_scaler = np.array([1 / 3048, 1 / r, 1])
        cl = cu = np.array([3048, r, 1])

        # variable bounds
        eta_l, eta_u = np.full((n_eta), 0.0), np.full((n_eta), 1.0)
        theta_l, theta_u = np.full((n_theta), np.deg2rad(0)), np.full((n_theta), np.deg2rad(15))
        tf_l, tf_u = np.array([1e3]), np.array([50000])
        AR_l, AR_u = np.array([5.0]), np.array([50.0])
        S_l, S_u = np.array([30.0]), np.array([150.0])
        fuel_l, fuel_u = np.array([300.0]), np.array([10000.0])
        xl = np.concatenate((eta_l, theta_l, tf_l, AR_l, S_l, fuel_l))
        xu = np.concatenate((eta_u, theta_u, tf_u, AR_u, S_u, fuel_u))

        jaxprob = JaxProblem(x0=x0, jax_obj=jax_obj, jax_con=jax_con, xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_scaler, c_scaler=c_scaler)
        # optimizer = IPOPT(jaxprob, solver_options={'max_iter': 500, 'tol': 1e-7}, turn_off_outputs=True)
        optimizer = SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-7}, turn_off_outputs=True)

        t1 = time.perf_counter()
        optimizer.solve()
        opt_time = time.perf_counter() - t1
        data.append(opt_time + (data[-1] if len(data)>0 else 0))


        # optimizer.print_results()

        ans = optimizer.results['x'] / x_scaler

        # update lists with subPi results
        eta_i = ans[:n_eta]
        theta_i = ans[n_eta:-4]
        tf_i = ans[-4]
        AR_i = ans[-3]
        S_i = ans[-2]
        fuel_i = ans[-1]

        v_init[subP] = np.concatenate((eta_i, theta_i, np.array([tf_i]), np.array([AR_i]), np.array([S_i]), np.array([fuel_i])))

        gc.collect()
        return v_init



    return subP_i