# from lifting_line_jax_3 import LiftingLine
from lifting_line_jax_4 import LiftingLine
from beam_jax import Beam, CSTube
import numpy as np
from crm_mesh import build_crm_mesh
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv
import jax
jax.config.update("jax_enable_x64", True)
from modopt import JaxProblem, SLSQP
from jax_b_splines import get_bspline_mtx, bspline_comp

# aero parameters
v_inf   = 210 # (Mach 0.7)
rho_atm = 0.4135 # (30 kft altitude)
q       = 0.5 * rho_atm * v_inf**2

# generate the CRM lifting line mesh
ns = 33 # num spanwise panels (must be odd)
crm_mesh = build_crm_mesh(ns=ns, span_cos_spacing=0)
# print("CRM mesh shape:", crm_mesh.shape)
# exit()

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
chord = 0.5 * (chord[:-1] + chord[1:])
beam_radius = 0.25 * chord / 2
print("Beam radius at each element:\n", beam_radius)
# exit()
E = 69e9
G = 26e9
rho_mat = 3000
m0 = 1e5
load_factor = 3
safety_factor = 1.5
tip_disp_target = 0.1


# instantiate the lifting line model
lifting_line = LiftingLine(le, te, v_inf, rho_atm)


n_twist_cp = 7
twist_bspline_mtx = get_bspline_mtx(n_twist_cp, ns)

n_thickness_cp = 10
thickness_bspline_mtx = get_bspline_mtx(n_thickness_cp, ns - 1)


def objective(x):

    twist_cp = x[:n_twist_cp]
    twist = bspline_comp(twist_bspline_mtx, twist_cp)

    sol = lifting_line.solve_lifting_line_model(twist)
    CD = sol["CD"]

    return CD


def constraints(x):

    twist_cp = x[:n_twist_cp]
    twist = bspline_comp(twist_bspline_mtx, twist_cp)
    thickness_cp = x[n_twist_cp:]
    thickness = bspline_comp(thickness_bspline_mtx, thickness_cp)

    sol = lifting_line.solve_lifting_line_model(twist)

    aero_forces = sol["F"] * load_factor * safety_factor
    F = jnp.zeros((ns, 6))
    F = F.at[:, :3].set(aero_forces)

    cs = CSTube(radius=beam_radius, thickness=thickness)
    beam = Beam(mesh=beam_mesh, E=E, G=G, rho=rho_mat,
                A=cs.area, J=cs.J, Iy=cs.Iy, Iz=cs.Iz, F=F, 
                fixed_nodes=[ns // 2])
    u = beam.solve()
    u = jnp.linalg.norm(u[:, :3], axis=1)
    right_tip_disp, left_tip_disp = u[-1], u[0]

    crm_weight = (beam.mass + m0) * 9.81

    lift = sol['CL'] * q * lifting_line.S

    con = jnp.zeros(3)
    con = con.at[0].set(left_tip_disp - tip_disp_target)
    con = con.at[1].set(right_tip_disp - tip_disp_target)
    con = con.at[2].set(lift - crm_weight)
    return con






# thickness0 = np.ones(ns - 1) * 0.002
thickness_cp0 = np.ones(n_thickness_cp) * 0.002
# twist0 = np.ones(ns) * np.deg2rad(5)
twist_cp0 = np.ones(n_twist_cp) * np.deg2rad(5)
x0 = np.concatenate([twist_cp0, thickness_cp0])

thickness_lower = np.ones(n_thickness_cp) * 0.001 # min gauge
# thickness_upper = np.ones(n_thickness_cp) * 0.3 # np.inf
# thickness_upper = beam_radius # max thickness is when the inner radius goes to zero

thickness_upper = np.ones(n_thickness_cp) * min(beam_radius) # max thickness is when the inner radius goes to zero

twist_lower = -1 * np.ones(n_twist_cp) * np.inf
twist_upper = np.ones(n_twist_cp) * np.inf
xl = np.concatenate([twist_lower, thickness_lower])
xu = np.concatenate([twist_upper, thickness_upper])

cl = np.concatenate([-np.inf * np.ones(2), np.zeros(1)])
cu = np.concatenate([ np.zeros(2),         np.zeros(1)])

c_scaler = np.array([10, 10, 1e-4])

# x_scaler = np.concatenate([10 * np.ones(n_twist_cp),       # twist scaler
#                            100 * np.ones(ns - 1)]) # thickness scaler
x_scaler = np.concatenate([10 * np.ones(n_twist_cp),       # twist scaler
                           100 * np.ones(n_thickness_cp)]) # thickness scaler

jaxprob = JaxProblem(x0=x0, jax_obj=objective, jax_con=constraints, 
                     cl=cl, cu=cu, xl=xl, xu=xu, x_scaler=x_scaler, c_scaler=c_scaler, o_scaler=1e2)

optimizer = SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-7}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()
x = optimizer.results['x'] / x_scaler

twist_cp = x[:n_twist_cp]
thickness_cp = x[n_twist_cp:]

twist = bspline_comp(twist_bspline_mtx, twist_cp)
thickness = bspline_comp(thickness_bspline_mtx, thickness_cp)

sol = lifting_line.solve_lifting_line_model(twist)
CD = sol["CD"]
CL = sol["CL"]
Gamma = sol["Gamma"]
forces = sol["F"]

print("CL:", CL)
print("CD:", CD)


fig, ax = plt.subplots(2, 2, figsize=(10, 6))
ax = ax.flatten()
ax[0].plot(lifting_line.y, twist, linewidth=2)
ax[0].plot(np.linspace(lifting_line.y[0], lifting_line.y[-1], n_twist_cp), twist_cp, marker='o', linestyle='--')
ax[0].set_title("Twist")
ax[0].grid()
ax[1].plot(lifting_line.y, Gamma, linewidth=2)
ax[1].set_title("Gamma")

panel_centers = 0.5 * (beam_mesh[:-1, 1] + beam_mesh[1:, 1])
thickness_cp_span = np.linspace(panel_centers[0], panel_centers[-1], n_thickness_cp)

ax[2].plot(panel_centers, thickness, linewidth=2)
ax[2].plot(thickness_cp_span, thickness_cp, marker='o', linestyle='--')
ax[2].grid()
ax[2].set_title("Thickness")

plt.show()


# fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# ax[0].plot(lifting_line.y, twist, linewidth=2)
# ax[0].scatter(np.linspace(lifting_line.y[0], lifting_line.y[-1], n_twist_cp), twist_cp, color='red')
# ax[0].set_title("Twist")
# ax[1].plot(lifting_line.y, Gamma, linewidth=2)
# ax[1].set_title("Gamma")
# plt.show()

# plotter = pv.Plotter()
# lifting_line.plot_3d(Gamma, forces, plotter)
# plotter.view_isometric()
# plotter.show()

# b = np.linalg.norm(te[0] - te[-1])
# plt.plot(np.linspace(-b/2, b/2, ns - 1), thickness)
# plt.xlabel('Spanwise Position')
# plt.ylabel('Thickness')
# plt.title('Thickness Distribution')
# plt.grid()
# plt.show()


# aero_forces = sol["F"] * load_factor * safety_factor
# F = jnp.zeros((ns, 6))
# F = F.at[:, :3].set(aero_forces)

# cs = CSTube(radius=beam_radius, thickness=thickness)
# beam = Beam(mesh=beam_mesh, E=E, G=G, rho=rho_mat,
#             A=cs.area, J=cs.J, Iy=cs.Iy, Iz=cs.Iz, F=F, 
#             fixed_nodes=[ns // 2])
# u = beam.solve()
# u = jnp.linalg.norm(u[:, :3], axis=1)
# right_tip_disp, left_tip_disp = u[-1], u[0]

# crm_mass = beam.mass + m0
# print("CRM Mass:", crm_mass)

# print("Left Tip Displacement:", left_tip_disp)
# print("Right Tip Displacement:", right_tip_disp)