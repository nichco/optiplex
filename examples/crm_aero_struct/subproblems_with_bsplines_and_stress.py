from lifting_line_jax_4 import LiftingLine
from beam_jax_2 import Beam, CSTube
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
from jax_b_splines import get_bspline_mtx, bspline_comp


# generate the CRM lifting line mesh
ns = 45 # num spanwise panels (must be odd)
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


n_twist_cp = 17
twist_bspline_mtx = get_bspline_mtx(n_twist_cp, ns)

n_thickness_cp = 10
thickness_bspline_mtx = get_bspline_mtx(n_thickness_cp, ns - 1)


def make_subproblem(subP, num, opt_time, samples):

    def subP_i(v_init: list,
               y: np.ndarray = None, # lagrange multipliers
               mu: float = 1, # penalty parameter
               ) -> list:
        
        print(f"Solving subproblem {subP}")

        alphas = [x_init_i[0] for x_init_i in v_init]
        twist_cps = [x_init_i[1:1 + n_twist_cp] for x_init_i in v_init]
        thickness_cps = [x_init_i[1 + n_twist_cp:] for x_init_i in v_init]

        def objective(x):
            alpha_i = x[0] # trim angle for this condition
            twist_cp_i = x[1:1 + n_twist_cp] # twist distribution for this condition
            thickness_cp_i = x[1 + n_twist_cp:] # thickness distribution for this condition

            alphas[subP] = alpha_i
            twist_cps[subP] = twist_cp_i
            thickness_cps[subP] = thickness_cp_i
            
            # twist_i = bspline_comp(twist_bspline_mtx, twist_cp_i)
            # rho_atm_i, v_inf_i = samples[subP]
            # effective_twist = twist_i + alpha_i # add the trim aoa to the twist distribution
            # ll = LiftingLine(le, te, v_inf_i, rho_atm_i)
            # sol = ll.solve_lifting_line_model(effective_twist)
            # obj = sol["CD"]

            obj = 0.0
            for j in range(num):
                rho_atm_j, v_inf_j = samples[j]
                # twist_j = bspline_comp(twist_bspline_mtx, twist_cps[j])
                twist_j = bspline_comp(twist_bspline_mtx, twist_cps[0])
                ll_j = LiftingLine(le, te, v_inf_j, rho_atm_j)
                sol_j = ll_j.solve_lifting_line_model(twist_j + alphas[j])
                obj += sol_j["CD"]

            obj = obj / num # minimize the average CD across all conditions
            obj += jnp.sum(jnp.array(alphas)**2) * 1e1

            delta_twist_cp = twist_cps[0][1:] - twist_cps[0][:-1] # variation in twist_cp
            obj += jnp.sum(delta_twist_cp**2) * 2e-2


            twist_cp_constraint = combo(twist_cps)
            thickness_cp_constraint = combo(thickness_cps)
            # c = jnp.concatenate((0.125*twist_cp_constraint, 2*thickness_cp_constraint))
            c = jnp.concatenate((0.125*twist_cp_constraint, 1*thickness_cp_constraint))

            # return 1e2 * obj + y.T @ c + 0.5 * mu * jnp.sum(c**2)
            return 1e2 * obj + y.T @ c + 0.5 * c.T @ jnp.diag(mu) @ c


        def constraints(x):

            alpha_i = x[0] # trim angle for this condition
            twist_cp_i = x[1:1 + n_twist_cp] # twist distribution for this condition
            thickness_cp_i = x[1 + n_twist_cp:] # thickness distribution for this condition

            twist_i = bspline_comp(twist_bspline_mtx, twist_cp_i)
            thickness_i = bspline_comp(thickness_bspline_mtx, thickness_cp_i)

            effective_twist = twist_i + alpha_i # add the trim aoa to the twist distribution

            rho_atm_i, v_inf_i = samples[subP]
            ll = LiftingLine(le, te, v_inf_i, rho_atm_i)
            sol = ll.solve_lifting_line_model(effective_twist)
            CL = sol["CL"]
            forces = sol["F"]

            forces *= load_factor * safety_factor
            F = jnp.zeros((ns, 6))
            F = F.at[:, :3].set(forces)

            cs = CSTube(radius=beam_radius, thickness=thickness_i)
            beam = Beam(mesh=beam_mesh, E=E, G=G, rho=rho_mat, cs=cs, F=F, fixed_nodes=[ns // 2])
            u = beam.solve()
            sigma = beam.recover_stress(u)
            sigma_mpa = sigma / 1e6
            rho = 2e-1
            max_sigma_mpa = jnp.log(jnp.sum(jnp.exp(rho * (sigma_mpa)))) / rho

            crm_weight = (beam.mass + m0) * 9.81

            q = 0.5 * rho_atm_i * v_inf_i**2
            lift = CL * q * ll.S

            con_i = jnp.zeros(2)
            con_i = con_i.at[0].set(max_sigma_mpa)
            con_i = con_i.at[1].set(lift - crm_weight)
            return con_i
        



        alpha_i_0 = alphas[subP]
        thickness_cp_i_0 = thickness_cps[subP]
        twist_cp_i_0 = twist_cps[subP]
        x0 = np.concatenate([np.array([alpha_i_0]), np.array(twist_cp_i_0), np.array(thickness_cp_i_0)])

        alpha_lower = -1 * np.ones(1) * np.deg2rad(5)
        alpha_upper = np.ones(1) * np.deg2rad(5)
        thickness_cp_lower = np.ones(n_thickness_cp) * 0.001 # min gauge
        thickness_cp_upper = np.ones(n_thickness_cp) * min(beam_radius) # max thickness is when the inner radius goes to zero
        twist_cp_lower = -1 * np.ones(n_twist_cp) * np.deg2rad(10)
        twist_cp_upper = np.ones(n_twist_cp) * np.deg2rad(10)
        xl = np.concatenate([alpha_lower, twist_cp_lower, thickness_cp_lower])
        xu = np.concatenate([alpha_upper, twist_cp_upper, thickness_cp_upper])

        cl_i = np.array([-np.inf, 0])
        cu_i = np.array([500, 0])
        c_i_scaler = np.array([1e-2, 1e-4])

        x_scaler = np.concatenate([np.array([100]),    # alpha scaler
                                   10 * np.ones(n_twist_cp),      # twist scaler
                                   10 * np.ones(n_thickness_cp)]) # thickness scaler
        
        jaxprob = JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints, 
                     cl=cl_i, cu=cu_i, xl=xl, xu=xu, x_scaler=x_scaler, c_scaler=c_i_scaler)

        optimizer = SLSQP(jaxprob, solver_options={'maxiter': 700, 'ftol': 1e-8}, turn_off_outputs=True)

        t0 = time.perf_counter()
        optimizer.solve()
        t1 = time.perf_counter()

        elapsed_time = t1 - t0
        opt_time.append(elapsed_time + (opt_time[-1] if len(opt_time)>0 else 0))

        # optimizer.print_results()
        x = optimizer.results['x'] / x_scaler

        alpha_i = x[0] # trim angle for this condition
        twist_cp_i = x[1:1 + n_twist_cp] # twist distribution for this condition
        thickness_cp_i = x[1 + n_twist_cp:] # thickness distribution for this condition

        ans = v_init.copy()
        ans[subP] = np.concatenate([np.array([alpha_i]), np.array(twist_cp_i), np.array(thickness_cp_i)])

        gc.collect()
        return ans


    return subP_i