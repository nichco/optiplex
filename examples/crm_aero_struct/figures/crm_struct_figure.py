import pyvista as pv
import numpy as np
from crm_mesh import build_crm_mesh

# plotter = pv.Plotter()
plotter = pv.Plotter(off_screen=True) # use this when saving images

ns       = 33
crm_mesh = build_crm_mesh(ns=ns, span_cos_spacing=0)
le_pts   = crm_mesh[0, :, :]
te_pts   = crm_mesh[1, :, :]

# Stack LE/TE points into a (2, ns, 3) array and build a structured surface
points = np.empty((2, ns, 3))
points[0, :, :] = le_pts
points[1, :, :] = te_pts

grid = pv.StructuredGrid()
grid.points = points.reshape(-1, 3)   # j (spanwise) varies fastest
grid.dimensions = [ns, 2, 1]          # matches x-fastest point ordering above

# Color each cell by the norm of its distance to the y=0 symmetry plane
cell_centers = grid.cell_centers().points
dist_to_symmetry = np.linalg.norm(cell_centers[:, 1:2], axis=1)  # |y| per cell
grid.cell_data["dist_to_symmetry"] = dist_to_symmetry

plotter.add_mesh(
    grid,
    scalars="dist_to_symmetry",
    cmap="plasma",          # root -> dark end, wingtips -> bright end
    show_edges=True,
    scalar_bar_args={"title": "Distance to Symmetry Plane"},
)

plotter.view_vector(vector=[0, 0, 1.5]) 
# plotter.show()
plotter.screenshot('crm_struct.png', transparent_background=True, window_size=[3000, 2000])