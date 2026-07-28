from lifting_line_jax_4 import LiftingLine
from beam import Beam, CSTube
import numpy as np
from crm_mesh import build_crm_mesh
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
from modopt import JaxProblem, SLSQP
from scipy.stats.qmc import LatinHypercube, scale
from jax_b_splines import get_bspline_mtx, bspline_comp
import tracemalloc
tracemalloc.start()

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


n_twist_cp = 15
twist_bspline_mtx = get_bspline_mtx(n_twist_cp, ns)

n_thickness_cp = 10
thickness_bspline_mtx = get_bspline_mtx(n_thickness_cp, ns - 1)



num = 3 # number of operating conditions
sampler = LatinHypercube(d=2, seed=42)
samples = scale(sampler.random(num), l_bounds=[0.4, 180], u_bounds=[0.6, 220])
# samples = [(0.4135, 210), (0.4135, 207)] # test solution a
# samples = [(0.4135, 210), (0.4135, 210)] # test solution b
# samples = [(0.4135, 210), (0.5, 191)] # test solution c
# samples = [(0.4226044, 191.2224312), (0.51414021, 206.05263942)] # test solution d
print(samples)




def objective(x):

    alphas = x[:num]
    twist_cp = x[num:num + n_twist_cp]

    drag_coefs = []
    for i in range(num):
        twist_i = bspline_comp(twist_bspline_mtx, twist_cp)
        twist_i = twist_i + alphas[i]
        rho_atm_i, v_inf_i = samples[i]
        ll = LiftingLine(le, te, v_inf_i, rho_atm_i)
        sol = ll.solve_lifting_line_model(twist_i)
        drag_coefs.append(sol["CD"])

    avg_drag_coef = jnp.mean(jnp.array(drag_coefs))
    obj = avg_drag_coef

    # obj += jnp.sum(alphas**2) * 1e1 # remove the differential flatness in the trim solution

    return 1e2 * obj


def constraints(x):

    alphas = x[:num]
    twist_cp = x[num:num + n_twist_cp]
    thickness_cp = x[num + n_twist_cp:]

    twist = bspline_comp(twist_bspline_mtx, twist_cp)
    thickness = bspline_comp(thickness_bspline_mtx, thickness_cp)

    cons = []

    for i in range(num):

        rho_atm, v_inf = samples[i]
        alpha_i = alphas[i]

        effective_twist = twist + alpha_i # add the trim aoa to the twist distribution

        lifting_line = LiftingLine(le, te, v_inf, rho_atm)
        sol = lifting_line.solve_lifting_line_model(effective_twist)
        CL = sol["CL"]
        forces = sol["F"]

        forces *= load_factor * safety_factor
        F = jnp.zeros((ns, 6))
        F = F.at[:, :3].set(forces)

        cs = CSTube(radius=beam_radius, thickness=thickness)
        beam = Beam(mesh=beam_mesh, E=E, G=G, rho=rho_mat, cs=cs, F=F, fixed_nodes=[ns // 2])
        u = beam.solve()
        sigma = beam.recover_stress(u)
        sigma_mpa = sigma / 1e6
        rho = 2e-1
        max_sigma_mpa = jnp.log(jnp.sum(jnp.exp(rho * (sigma_mpa)))) / rho

        crm_weight = (beam.mass + m0) * 9.81

        q = 0.5 * rho_atm * v_inf**2
        lift = CL * q * lifting_line.S

        con_i = jnp.zeros(2)
        con_i = con_i.at[0].set(max_sigma_mpa)
        con_i = con_i.at[1].set(lift - crm_weight)

        cons.append(con_i)

    cons = jnp.array(cons).flatten()
    # add a constraint alphas[0]
    cons = jnp.concatenate((cons, jnp.array([alphas[0]])))

    return jnp.array(cons).flatten()






alpha0 = np.zeros(num) # start with zero trim aoa for all conditions
thickness_cp0 = np.ones(n_thickness_cp) * 0.002
twist_cp0 = np.ones(n_twist_cp) * np.deg2rad(5)
x0 = np.concatenate([alpha0, twist_cp0, thickness_cp0])

alpha_lower = -1 * np.ones(num) * np.deg2rad(5)
alpha_upper = np.ones(num) * np.deg2rad(5)
thickness_lower = np.ones(n_thickness_cp) * 0.001 # min gauge
thickness_upper = np.ones(n_thickness_cp) * min(beam_radius) # max thickness is when the inner radius goes to zero
twist_lower = -1 * np.ones(n_twist_cp) * np.deg2rad(10)
twist_upper = np.ones(n_twist_cp) * np.deg2rad(10)
xl = np.concatenate([alpha_lower, twist_lower, thickness_lower])
xu = np.concatenate([alpha_upper, twist_upper, thickness_upper])



cl_i = np.array([-np.inf, 0])
cu_i = np.array([500, 0])

cl = np.concatenate([cl_i for _ in range(num)])
cu = np.concatenate([cu_i for _ in range(num)])

# append 0 to the constraint bounds for alphas[0]
cl = np.concatenate((cl, jnp.array([0])))
cu = np.concatenate((cu, jnp.array([0])))

c_i_scaler = np.array([1e-2, 1e-4])
c_scaler = np.concatenate([c_i_scaler for _ in range(num)])

# append 1 to the constraint scaler for alphas[0]
c_scaler = np.concatenate((c_scaler, jnp.array([1])))


x_scaler = np.concatenate([100 * np.ones(num),      # alpha scaler
                           10 * np.ones(n_twist_cp),       # twist scaler
                           10 * np.ones(n_thickness_cp)]) # thickness scaler





jaxprob = JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints, 
                     cl=cl, cu=cu, xl=xl, xu=xu, x_scaler=x_scaler, c_scaler=c_scaler)

optimizer = SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-8}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()
x = optimizer.results['x'] / x_scaler

# print peak memory usage
_, peak = tracemalloc.get_traced_memory()
print(f"Peak: {peak / 10**6}MB")
tracemalloc.stop()

alphas = x[:num]
twist_cp = x[num:num + n_twist_cp]
thickness_cp = x[num + n_twist_cp:]

twist = bspline_comp(twist_bspline_mtx, twist_cp)
thickness = bspline_comp(thickness_bspline_mtx, thickness_cp)

print("Trim aoa (deg):", np.rad2deg(alphas))



y_aero = crm_mesh[0, :, 1] # spanwise y-coordinates from the mesh
fig, ax = plt.subplots(1, 2, figsize=(8, 2), constrained_layout=True)
ax[0].plot(y_aero, twist, linewidth=2)
ax[0].plot(np.linspace(y_aero[0], y_aero[-1], n_twist_cp), twist_cp, marker='o', linestyle='--')
ax[0].set_title("Twist")
ax[0].set_xlabel("Spanwise location (m)")
ax[0].set_ylabel("Twist (rad)")

panel_centers = 0.5 * (beam_mesh[:-1, 1] + beam_mesh[1:, 1])
thickness_cp_span = np.linspace(panel_centers[0], panel_centers[-1], n_thickness_cp)

ax[1].plot(panel_centers, thickness, linewidth=2)
ax[1].plot(thickness_cp_span, thickness_cp, marker='o', linestyle='--')
# ax[1].grid()
ax[1].set_title("Thickness")
ax[1].set_xlabel("Spanwise location (m)")
ax[1].set_ylabel("Thickness (m)")

plt.show()

np.savez('examples/crm_aero_struct/solution_N3_V2.npz', alphas=alphas, twist_cp=twist_cp, thickness_cp=thickness_cp, samples=samples)