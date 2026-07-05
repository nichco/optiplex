import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from modopt import JaxProblem, SLSQP
import gc
from optiplex import combo
import warnings
warnings.filterwarnings("ignore")
import time


# parameters
n = 30
dt = 2 / n
mc = 2
d = 0.8
x0 = np.array([0, np.pi, 0, 0])
xf = np.array([d, 0, 0, 0])

ni = 0
ni += 4 * n # num state vars
ni += n # num control vars
# l and mp are the final two vars


def make_subproblem(subP, N, opt_time, samples):

    def subP_i(v_init: list,
               y: np.ndarray = None, # lagrange multipliers
               mu: float = 1, # penalty parameter
               ) -> list:
        
        g_i = samples[subP, 0]
        mu_cart_i = samples[subP, 1]
        mu_pole_i = samples[subP, 2]
        
        # l_i = v_i[0]
        # mp_i = v_i[1]
        # x_i = v_i[2:2 + 4 * n].reshape((4, n))
        # u_i = v_i[2 + 4 * n:]

        l_list = [x_init_i[0] for x_init_i in v_init]
        mp_list = [x_init_i[1] for x_init_i in v_init]
        state_list = [x_init_i[2:2 + 4 * n] for x_init_i in v_init]
        u_list = [x_init_i[2 + 4 * n:] for x_init_i in v_init]

        def jax_obj(v_i):

            l_list[subP] = v_i[0]
            mp_list[subP] = v_i[1]
            u_list[subP] = v_i[2 + 4 * n:]

            j = 0
            for i in range(N):
                u_i = u_list[i]
                j_i = 0.5 * dt * jnp.sum(u_i[:-1]**2 + u_i[1:]**2)
                j += j_i

            l_constraint = combo(l_list)
            mp_constraint = combo(mp_list)
            c = jnp.concatenate((l_constraint, mp_constraint))

            return 1e-2 * j + y.T @ c + 0.5 * mu * jnp.sum(c**2)
            # return 1e-2 * j + y.T @ c + 0.5 * c.T @ jnp.diag(mu) @ c
            # return j / N
        
        
        def jax_con(v_i):

            # v_i = [l_i, mp_i, x_i, u_i]
            # l_i: scalar, mp_i: scalar, x_i: (4, n), u_i: (n,)
            l_i = v_i[0]
            mp_i = v_i[1]
            x_i = v_i[2:2 + 4 * n].reshape((4, n))
            u_i = v_i[2 + 4 * n:]

            l_hat = l_i / 2

            theta = x_i[1, :]
            dx = x_i[2, :]
            dtheta = x_i[3, :]
            cart_force = u_i

            ddx_num = mp_i * g_i * jnp.sin(theta) * jnp.cos(theta) - (7/3) *\
                (cart_force + mp_i * l_hat * dtheta**2 * jnp.sin(theta) -\
                    mu_cart_i * dx) - (mu_pole_i * dtheta * jnp.cos(theta) / l_hat)
            
            ddx_den = mp_i * jnp.cos(theta)**2 - (7/3) * (mc + mp_i)

            ddx = ddx_num / ddx_den

            ddtheta = 3 * (g_i * jnp.sin(theta) - ddx * jnp.cos(theta) - (mu_pole_i * dtheta / (mp_i * l_hat))) / (7 * l_hat)

            f = jnp.vstack((dx, dtheta, ddx, ddtheta)) # (4, n)

            # trapezoidal collocation constraints
            r = x_i[:, 1:] - x_i[:, :-1] - 0.5 * dt * (f[:, 1:] + f[:, :-1]) # (4, n-1)
            r = r.flatten()

            # initial condition constraints
            x0 = x_i[:, 0] # 

            # terminal constraints
            xf = x_i[:, n - 1]

            return jnp.concatenate((r, x0, xf))
        

        
        cl_r = np.zeros((4 * (n - 1)))
        cu_r = np.zeros((4 * (n - 1)))
        cl = np.concatenate((cl_r, x0, xf))
        cu = np.concatenate((cu_r, x0, xf))

        r_scaler = np.ones(4 * (n - 1)) * 1e1
        x0_scaler = np.ones(4) * 1
        xf_scaler = np.ones(4) * 1
        c_scaler = np.concatenate((r_scaler, x0_scaler, xf_scaler))

        state_u = np.ones((4, n)) * np.inf
        state_l = np.ones((4, n)) * -np.inf
        l_u = np.array([5])
        l_l = np.array([0.1])
        mp_u = np.array([3])
        mp_l = np.array([0.1])
        u_u = np.ones((n)) * 50
        u_l = np.ones((n)) * -50
        xl = np.concatenate((l_l, mp_l, state_l.flatten(), u_l))
        xu = np.concatenate((l_u, mp_u, state_u.flatten(), u_u))

        l_0 = np.array([0.5])
        mp_0 = np.array([0.4])
        q1_0 = np.linspace(0, d, n)
        q2_0 = np.linspace(np.pi, 0, n)
        q3_0 = np.zeros(n)
        q4_0 = np.zeros(n)
        state_0 = np.vstack((q1_0, q2_0, q3_0, q4_0)).flatten()
        u_0 = np.zeros(n)
        v0 = np.concatenate((l_0, mp_0, state_0, u_0))

        l_scaler = np.ones(1)
        mp_scaler = np.ones(1)
        state_scaler = np.ones(4 * n)
        u_scaler = np.ones(n) * 1e-1
        x_scaler = np.concatenate((l_scaler, mp_scaler, state_scaler, u_scaler))

        o_scaler = 1 # scaler accounted for in jax_obj()

        jaxprob = JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=cl, cu=cu, 
                            xl=xl, xu=xu, o_scaler=o_scaler, c_scaler=c_scaler, x_scaler=x_scaler)
        optimizer = SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-9}, turn_off_outputs=True)
        optimizer.solve()
        # optimizer.print_results()
        ans = optimizer.results['x'] / x_scaler

        l_i = ans[0]
        mp_i = ans[1]
        x_i = ans[2:2 + 4 * n].reshape((4, n))
        u_i = ans[2 + 4 * n:]

        result = v_init.copy()
        result[subP] = ans

        gc.collect()

        return result
    
    return subP_i