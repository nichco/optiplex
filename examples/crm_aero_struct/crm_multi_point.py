# from lifting_line_jax_3 import LiftingLine
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


# generate the CRM lifting line mesh
ns = 33 # num spanwise panels (must be odd)
crm_mesh = build_crm_mesh(ns=ns, span_cos_spacing=0)

# generate a beam mesh from the CRM lifting line mesh
le = crm_mesh[0, :, :]
te = crm_mesh[1, :, :]
beam_mesh = (le + te) / 2.0

# # plot both meshes
# plt.scatter(crm_mesh[:, :, 0], crm_mesh[:, :, 1], color='tab:blue')
# plt.scatter(beam_mesh[:, 0], beam_mesh[:, 1], color='tab:orange')
# plt.axis('equal')
# plt.show()

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




num = 2 # number of operating conditions
sampler = LatinHypercube(d=2, seed=42)
samples = scale(sampler.random(num), l_bounds=[0.4, 180], u_bounds=[0.6, 220])
# samples = [(0.4135, 210), (0.4135, 207)]
print(samples)


def condition(rho_atm, v_inf, alpha, twist, thickness):

    twist = twist + alpha# add the trim aoa to the twist distribution

    # instantiate the lifting line model
    lifting_line = LiftingLine(le, te, v_inf, rho_atm)
    sol = lifting_line.solve_lifting_line_model(twist)
    CD = sol["CD"]
    CL = sol["CL"]
    forces = sol["F"]

    forces *= load_factor * safety_factor
    F = jnp.zeros((ns, 6))
    F = F.at[:, :3].set(forces)

    cs = CSTube(radius=beam_radius, thickness=thickness)
    beam = Beam(mesh=beam_mesh, E=E, G=G, rho=rho_mat, cs=cs, F=F, fixed_nodes=[ns // 2])
    u = beam.solve()
    u = jnp.linalg.norm(u[:, :3], axis=1)
    right_tip_disp, left_tip_disp = u[-1], u[0]

    crm_weight = (beam.mass + m0) * 9.81

    q = 0.5 * rho_atm * v_inf**2
    lift = CL * q * lifting_line.S

    con = jnp.zeros(3)
    con = con.at[0].set(left_tip_disp - tip_disp_target)
    con = con.at[1].set(right_tip_disp - tip_disp_target)
    con = con.at[2].set(lift - crm_weight)

    return CD, con # (objective, constraints)




def objective(x):

    alphas = x[:num]
    twist = x[num:num + ns]

    obj = 0.0
    for i in range(num):
        rho_atm_i, v_inf_i = samples[i]
        effective_twist = twist + alphas[i]
        ll = LiftingLine(le, te, v_inf_i, rho_atm_i)
        sol = ll.solve_lifting_line_model(effective_twist)
        obj += sol["CD"]

    obj = obj / num # minimize the average CD across all conditions

    obj += jnp.sum(alphas**2) * 1e1 # remove the differential flatness in the trim solution

    return obj


def constraints(x):

    alphas = x[:num]
    twist = x[num:num + ns]
    thickness = x[num + ns:]

    cons = []

    for i in range(num):

        rho_atm, v_inf = samples[i]
        alpha_i = alphas[i]

        _, con_i = condition(rho_atm, v_inf, alpha_i, twist, thickness)

        cons.append(con_i)

    return jnp.array(cons).flatten()






alpha0 = np.zeros(num) # start with zero trim aoa for all conditions
thickness0 = np.ones(ns - 1) * 0.002
twist0 = np.ones(ns) * np.deg2rad(5)
x0 = np.concatenate([alpha0, twist0, thickness0])

alpha_lower = -1 * np.ones(num) * np.deg2rad(5)
alpha_upper = np.ones(num) * np.deg2rad(10)
thickness_lower = np.ones(ns - 1) * 0.001 # min gauge
# thickness_upper = np.ones(ns - 1) * 0.3 # np.inf
thickness_upper = beam_radius # max thickness is when the inner radius goes to zero
# twist_lower = -1 * np.ones(ns) * np.inf
# twist_upper = np.ones(ns) * np.inf
twist_lower = -1 * np.ones(ns) * np.deg2rad(15)
twist_upper = np.ones(ns) * np.deg2rad(15)
xl = np.concatenate([alpha_lower, twist_lower, thickness_lower])
xu = np.concatenate([alpha_upper, twist_upper, thickness_upper])



cl_i = np.concatenate([-np.inf * np.ones(2), np.zeros(1)])
cu_i = np.concatenate([ np.zeros(2),         np.zeros(1)])

cl = np.concatenate([cl_i for _ in range(num)])
cu = np.concatenate([cu_i for _ in range(num)])

c_i_scaler = np.array([10, 10, 1e-4])
c_scaler = np.concatenate([c_i_scaler for _ in range(num)])


x_scaler = np.concatenate([100 * np.ones(num),      # alpha scaler
                           10 * np.ones(ns),       # twist scaler
                           10 * np.ones(ns - 1)]) # thickness scaler





jaxprob = JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints, 
                     cl=cl, cu=cu, xl=xl, xu=xu, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=1e2)

optimizer = SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-8}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()
x = optimizer.results['x'] / x_scaler

alphas = x[:num]
twist = x[num:num + ns]
thickness = x[num + ns:]

print("Trim aoa (deg):", np.rad2deg(alphas))



y_aero = crm_mesh[0, :, 1] # spanwise y-coordinates from the mesh
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(y_aero, twist, linewidth=2)
ax[0].set_title("Twist")

b = np.linalg.norm(te[0] - te[-1])
ax[1].plot(np.linspace(-b/2, b/2, ns - 1), thickness, linewidth=2)
ax[1].set_title("Thickness")
plt.show()




# save the solution to an npz file
# np.savez('test_solution_lhs_num_100.npz', alphas=alphas, twist=twist, thickness=thickness, samples=samples)




# lhs num 2 time: 9.374
# lhs num 5 time: 14.595
# lhs num 10 time: 19.39
# lhs num 100 time: 444.273