from lifting_line_jax_3 import LiftingLine
import numpy as np
from crm_mesh import build_crm_mesh
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv

v_inf  = 300.0
rho    = 1.0

ns = 33 # num spanwise panels (must be odd)
crm_mesh = build_crm_mesh(ns=ns, span_cos_spacing=0)

plt.scatter(crm_mesh[:, :, 0], crm_mesh[:, :, 1])
plt.axis('equal')
plt.show()

le = crm_mesh[0, :, :]
te = crm_mesh[1, :, :]
print('le:', le)
print('te:', te)

lifting_line = LiftingLine(le, te, v_inf, rho)

twist = jnp.ones(ns) * jnp.deg2rad(5)

sol = lifting_line.solve_lifting_line_model(twist)
CD = sol["CD"]
CL = sol["CL"]
Gamma = sol["Gamma"]
forces = sol["F"]

print("CL:", CL)
print("CD:", CD)

print('forces shape:', forces.shape)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(lifting_line.y, twist, linewidth=2)
ax[0].set_title("Twist")
ax[1].plot(lifting_line.y, Gamma, linewidth=2)
ax[1].set_title("Gamma")
plt.show()

plotter = pv.Plotter()
lifting_line.plot_3d(Gamma, forces, plotter)
plotter.view_isometric()
plotter.show()