import pyvista as pv
import pickle
import numpy as np

# plotter = pv.Plotter()
plotter = pv.Plotter(off_screen=True) # use this when saving images

# vlm mesh
with open("examples/crm_aero_struct/crm_mesh_21x6.pkl", "rb") as f:
    crm_mesh = pickle.load(f)
    # crm_mesh = crm_mesh[0] * 25.4 + np.array([-100, 0, -100])
    crm_mesh = crm_mesh[0] * 25.4 + np.array([-0, 0, 110])
    nc, ns, _ = crm_mesh.shape
    point_cloud = pv.PolyData(crm_mesh.reshape(-1, 3))
    # plotter.add_mesh(point_cloud, color="red", point_size=6, render_points_as_spheres=True)

# vlm pressure
with open("examples/crm_aero_struct/crm_panel_pressure.pkl", "rb") as f:
    data = pickle.load(f)



# flatten and scale points
points = crm_mesh.reshape(-1, 3)

# build quad faces for a structured (nc-1)×(ns-1) panel mesh
faces = []
for i in range(nc - 1):
    for j in range(ns - 1):
        # indices of the four corner nodes, ordered consistently
        p0 = i   * ns + j
        p1 = (i+1) * ns + j
        p2 = (i+1) * ns + (j+1)
        p3 = i   * ns + (j+1)
        faces.extend([4, p0, p1, p2, p3])
faces = np.array(faces, dtype=np.int64)

# create the PyVista mesh
mesh = pv.PolyData(points, faces)

# attach the data as cell scalars
mesh.cell_data["pressure"] = data.flatten()

smooth_mesh = mesh.cell_data_to_point_data()

plotter.add_mesh(mesh, cmap="viridis", show_edges=True, lighting=True, smooth_shading=False, show_scalar_bar=False,)
# plotter.add_mesh(smooth_mesh, cmap="viridis", show_edges=True, lighting=True, show_scalar_bar=False, edge_color='black')



# plot a point at 0, 0, 0 for reference
# plotter.add_mesh(pv.Sphere(radius=100), color="red", show_edges=False)


plotter.view_vector(vector=[0, 0, 1.5]) 
# plotter.show()
plotter.screenshot('crm_aero.png', transparent_background=True, window_size=[3000, 2000])