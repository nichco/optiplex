# from lifting_line_jax_3 import LiftingLine
from lifting_line_jax_4 import LiftingLine
from beam_jax import Beam, CSTube
import numpy as np
from crm_mesh import build_crm_mesh
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from modopt import JaxProblem, SLSQP
import gc
from optiplex import combo
import warnings
warnings.filterwarnings("ignore")
import time


# generate the CRM lifting line mesh
ns = 33 # num spanwise panels (must be odd)
crm_mesh = build_crm_mesh(ns=ns, span_cos_spacing=0)

# generate a beam mesh from the CRM lifting line mesh
le = crm_mesh[0, :, :]
te = crm_mesh[1, :, :]
beam_mesh = (le + te) / 2.0

# beam model parameters
chord = np.linalg.norm(te - le, axis=1)
# interpolate chord on a per-element basis (ns - 1)
interp_chord = 0.5 * (chord[:-1] + chord[1:])
beam_radius = 0.25 * interp_chord / 2
E, G = 69e9, 26e9
rho_mat = 3000
m0 = 1e5
load_factor = 3
safety_factor = 1.5
tip_disp_target = 0.1



# def make_subproblem(subP, rho_atm_i, v_inf_i, num, si, cs, opt_time, samples):
def make_subproblem(subP, num, cs, opt_time, samples):

    def subP_i(v_init: list,
               y: np.ndarray = None, # lagrange multipliers
               mu: float = 1, # penalty parameter
               ) -> list:
        
        print(f"Solving subproblem {subP}")

        alphas = []
        twists = []
        thicknesses = []
        for i in range(num):
            x_init_i = v_init[i]
            alphas.append(x_init_i[0])
            twists.append(x_init_i[1:1 + ns])
            thicknesses.append(x_init_i[1 + ns:])


        def objective(x):
            alpha_i = x[0] # trim angle for this condition
            twist_i = x[1:1 + ns] # twist distribution for this condition
            thickness_i = x[1 + ns:] # thickness distribution for this condition

            alpha_list = alphas.copy()
            twist_list = twists.copy()
            thickness_list = thicknesses.copy()

            # alphas[subP] = alpha_i
            # twists[subP] = twist_i
            # thicknesses[subP] = thickness_i
            alpha_list[subP] = alpha_i
            twist_list[subP] = twist_i
            thickness_list[subP] = thickness_i

            # effective_twist = twist_i + alpha_i # add the trim aoa to the twist distribution

            # ll = LiftingLine(le, te, v_inf_i, rho_atm_i)
            # sol = ll.solve_lifting_line_model(effective_twist)
            # CD = sol["CD"]
            obj = 0.0
            for i in range(num):
                rho_atm_i, v_inf_i = samples[i]
                effective_twist = twist_list[i] + alpha_list[i]
                ll = LiftingLine(le, te, v_inf_i, rho_atm_i)
                sol = ll.solve_lifting_line_model(effective_twist)
                obj += sol["CD"]

            obj = obj / num # minimize the average CD across all conditions
            obj += jnp.sum(jnp.array(alpha_list)**2) * 1e-2 # remove the differential flatness in the trim solution

            twist_constraint = combo(twist_list) # modified combo to remove one pair
            thickness_constraint = combo(thickness_list) # modified combo to remove one pair
        
            c = jnp.concatenate((twist_constraint, thickness_constraint)) * cs

            # L = 1e2 * obj #+ y.T @ c + 0.5 * mu * jnp.sum(c**2)
            L = 1e2 * obj + y.T @ c + 0.5 * mu * jnp.sum(c**2)
            return L


        def constraints(x):

            alpha_i = x[0] # trim angle for this condition
            twist_i = x[1:1 + ns] # twist distribution for this condition
            thickness_i = x[1 + ns:] # thickness distribution for this condition

            effective_twist = twist_i + alpha_i # add the trim aoa to the twist distribution

            rho_atm_i, v_inf_i = samples[subP]
            ll = LiftingLine(le, te, v_inf_i, rho_atm_i)
            sol = ll.solve_lifting_line_model(effective_twist)
            CD = sol["CD"]
            CL = sol["CL"]
            forces = sol["F"]

            forces *= load_factor * safety_factor
            F = jnp.zeros((ns, 6))
            F = F.at[:, :3].set(forces)

            cs = CSTube(radius=beam_radius, thickness=thickness_i)
            beam = Beam(mesh=beam_mesh, E=E, G=G, rho=rho_mat,
                        A=cs.area, J=cs.J, Iy=cs.Iy, Iz=cs.Iz, F=F, 
                        fixed_nodes=[ns // 2])
            u = beam.solve()
            u = jnp.linalg.norm(u[:, :3], axis=1)
            right_tip_disp, left_tip_disp = u[-1], u[0]

            crm_weight = (beam.mass + m0) * 9.81

            q = 0.5 * rho_atm_i * v_inf_i**2
            lift = CL * q * ll.S

            con_i = jnp.zeros(3)
            con_i = con_i.at[0].set(left_tip_disp - tip_disp_target)
            con_i = con_i.at[1].set(right_tip_disp - tip_disp_target)
            con_i = con_i.at[2].set(lift - crm_weight)

            return con_i
        



        alpha_i_0 = alphas[subP]
        thickness_i_0 = thicknesses[subP]
        twist_i_0 = twists[subP]
        x0 = np.concatenate([np.array([alpha_i_0]), np.array(twist_i_0), np.array(thickness_i_0)])

        alpha_lower = -1 * np.ones(1) * np.deg2rad(5)
        alpha_upper = np.ones(1) * np.deg2rad(10)
        thickness_lower = np.ones(ns - 1) * 0.001 # min gauge
        thickness_upper = beam_radius # max thickness is when the inner radius goes to zero
        twist_lower = -1 * np.ones(ns) * np.deg2rad(15)
        twist_upper = np.ones(ns) * np.deg2rad(15)
        xl = np.concatenate([alpha_lower, twist_lower, thickness_lower])
        xu = np.concatenate([alpha_upper, twist_upper, thickness_upper])

        cl = np.concatenate([-np.inf * np.ones(2), np.zeros(1)])
        cu = np.concatenate([ np.zeros(2),         np.zeros(1)])

        c_scaler = np.array([10, 10, 1e-4])


        x_scaler = np.concatenate([np.array([100]),    # alpha scaler
                                   10 * np.ones(ns),      # twist scaler
                                   10 * np.ones(ns - 1)]) # thickness scaler
        
        jaxprob = JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints, 
                     cl=cl, cu=cu, xl=xl, xu=xu, x_scaler=x_scaler, c_scaler=c_scaler)

        optimizer = SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-9}, turn_off_outputs=True)

        t0 = time.perf_counter()
        optimizer.solve()
        t1 = time.perf_counter()

        elapsed_time = t1 - t0
        opt_time.append(elapsed_time + (opt_time[-1] if len(opt_time)>0 else 0))

        # optimizer.print_results()
        x = optimizer.results['x'] / x_scaler

        alpha_i = x[0] # trim angle for this condition
        twist_i = x[1:1 + ns] # twist distribution for this condition
        thickness_i = x[1 + ns:] # thickness distribution for this condition

        # alphas[subP] = alpha_i
        # twists[subP] = twist_i
        # thicknesses[subP] = thickness_i

        # v_init[0] = alphas
        # v_init[1] = twists
        # v_init[2] = thicknesses

        ans_i = np.concatenate([np.array([alpha_i]), np.array(twist_i), np.array(thickness_i)])
        # v_init[subP] = ans_i
        ans = v_init.copy()
        ans[subP] = ans_i

        # gc.collect()
        return ans



    return subP_i