import jax.numpy as jnp
import jax
import numpy as np
from modopt import JaxProblem, SLSQP, IPOPT
jax.config.update("jax_enable_x64", True)
from vanilla_midpoint_rule import jax_midpoint
from mass_fun import _mass
from dynamics import f
from optiplex import combo
from plot import plot_trajectory

def make_sub_problem(mission_range: float, ind: int):
    # Define the sub-problem for a given mission range
    def sub_problem(v_init: list,
                    y: np.ndarray = None, # lagrange multipliers
                    mu: float = 1, # penalty coefficient
                    ) -> list:
        
        print('**********IND IND IND: ', ind)
        
        eta11 = v_init[0]
        theta11 = v_init[1]
        tf11 = v_init[2]
        b11= v_init[3]
        eta22 = v_init[4]
        theta22 = v_init[5]
        tf22 = v_init[6]
        b22= v_init[7]

        eta_list = [eta11, eta22] # need to expand for changing N
        theta_list = [theta11, theta22] # need to expand for changing N
        tf_list = [tf11, tf22] # need to expand for changing N
        b_list = [b11, b22] # need to expand for changing N

        t0 = 0.0
        nu = 60 # control n
        num_steps = 8000 # num steps

        # variable scaling
        eta_scale = 1.0
        theta_scale = 1e1
        tf_scale = 1e-3
        b_scale = 1e-1

        # y0 for ODE solver
        h0 = 3048.0 # m (10,000 ft)
        r0 = 0.0 # m
        v0 = 147.6 # m/s
        gamma0 = 0.0 # rad

        # terminal conditions
        hf = 3048.0 # m (10,000 ft)
        rf = mission_range # m (fixed range)
        vf = 147.6 # m/s (final velocity)

        payload_mass = 10000 # (kg)
        not_wing_mass = 20000 # (kg)
        mean_aerodynamic_chord = 3 # (m)

        def jax_obj(d):

            # scaled variables
            eta = d[:nu]
            theta = d[nu:-2]
            tf = d[-2]
            b = d[-1]

            # unscale variables
            eta = eta / eta_scale
            theta = theta / theta_scale
            tf = tf / tf_scale
            b = b / b_scale

            wing_area = b * mean_aerodynamic_chord
            m0 = _mass(payload_mass, not_wing_mass, wing_area, b)

            args = [eta, theta, tf, wing_area]

            h = tf / num_steps
            y0 = jnp.array([h0, r0, v0, gamma0, m0])
            # sol = jax_rk4(f, t0=t0, y0=y0, h=h, n=num_steps, args=args)
            sol = jax_midpoint(f, t0=t0, y0=y0, h=h, n=num_steps, args=args)

            m = sol[:, 4].ravel()
            m_f = m[-1]

            fuel_used = m0 - m_f


            b_list[ind] = b
            c_b = combo(b_list) # consensus for b
            # c = jnp.concatenate((c_b))
            c = c_b

            # return 1e-3 * fuel_used
            return 1e-3 * fuel_used + y.T @ c + mu * jnp.sum(c**2)


        def jax_con(d):

            # scaled variables
            eta = d[:nu]
            theta = d[nu:-2]
            tf = d[-2]
            b = d[-1]

            # unscale variables
            eta = eta / eta_scale
            theta = theta / theta_scale
            tf = tf / tf_scale
            b = b / b_scale

            wing_area = b * mean_aerodynamic_chord
            m0 = _mass(payload_mass, not_wing_mass, wing_area, b)

            args = [eta, theta, tf, wing_area]

            h = tf / num_steps
            y0 = jnp.array([h0, r0, v0, gamma0, m0])
            # sol = jax_rk4(f, t0=t0, y0=y0, h=h, n=num_steps, args=args)
            sol = jax_midpoint(f, t0=t0, y0=y0, h=h, n=num_steps, args=args)

            # solution
            h = sol[:, 0].ravel()
            r = sol[:, 1].ravel()
            v = sol[:, 2].ravel()
            gamma = sol[:, 3].ravel()
            m = sol[:, 4].ravel()

            # terminal constraints
            hf_constraint = h[-1] - hf
            rf_constraint = r[-1] - rf
            vf_constraint = v[-1] - vf

            con = jnp.zeros((3))
            con = con.at[0].set(hf_constraint * 1e-3)
            con = con.at[1].set(rf_constraint * 1e-6)
            con = con.at[2].set(vf_constraint * 1e-2)

            return con
        
        # variable bounds for ModOpt
        eta_l = np.full((nu), 0.0 * eta_scale)
        eta_u = np.full((nu), 1.0 * eta_scale)

        theta_l = np.full((nu), np.deg2rad(-10) * theta_scale)
        theta_u = np.full((nu), np.deg2rad(15) * theta_scale)

        tf_l = np.array([1e3]) * tf_scale
        tf_u = np.array([np.inf])

        b_l = np.array([10.0]) * b_scale
        b_u = np.array([50.0]) * b_scale

        xl = np.concatenate((eta_l, theta_l, tf_l, b_l))
        xu = np.concatenate((eta_u, theta_u, tf_u, b_u))


        # eta0 = np.linspace(0.6, 0.2, nu) * eta_scale
        # theta0 = np.linspace(np.deg2rad(3), np.deg2rad(1), nu) * theta_scale
        # tf0 = np.array([7000.0]) * tf_scale
        # b0 = np.array([32.0]) * b_scale
        # x0 = np.concatenate((eta0, theta0, tf0, b0))
        x0 = np.concatenate((eta_list[ind] * eta_scale, theta_list[ind] * theta_scale, tf_list[ind] * tf_scale, b_list[ind] * b_scale))



        jaxprob = JaxProblem(x0=x0, jax_obj=jax_obj, jax_con=jax_con, 
                            order=1, xl=xl, xu=xu, cl=0., cu=0.)


        # optimizer = SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-6}, turn_off_outputs=True)
        optimizer = IPOPT(jaxprob, solver_options={'max_iter': 150, 'tol': 1e-5}, turn_off_outputs=True)
        optimizer.solve()
        optimizer.print_results()


        x = optimizer.results['x']
        eta = x[:nu] / eta_scale
        theta = x[nu:-2] / theta_scale
        tf = x[-2] / tf_scale
        b = x[-1] / b_scale

        # plot_trajectory(h, r, v, gamma, m, eta, theta, tf)

        # start = ind * 4
        # v_init[start] = eta.flatten()
        # v_init[start + 1] = theta.flatten()
        # v_init[start + 2] = tf.flatten()
        # v_init[start + 3] = b.flatten()
        eta_list[ind] = eta.flatten()
        theta_list[ind] = theta.flatten()
        tf_list[ind] = tf.flatten()
        b_list[ind] = b.flatten()

        v_post = []
        for i in range(2):
            v_post.append(eta_list[i])
            v_post.append(theta_list[i])
            v_post.append(tf_list[i])
            v_post.append(b_list[i])

        return v_post

    return sub_problem