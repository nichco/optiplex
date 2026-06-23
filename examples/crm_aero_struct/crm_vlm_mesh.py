import numpy as np
import matplotlib.pyplot as plt
import pickle

# This is the jig CRM shape from the actual aircraft scale, not
# the wind tunnel model.
# eta, xle, yle, zle, twist, chord
# Info taken from AIAA paper 2008-6919 by Vassberg
raw_crm_points = np.array([
                [0.0, 904.294, 0.0, 174.126, 6.7166, 536.181],  # 0
                [0.1, 989.505, 115.675, 175.722, 4.4402, 468.511],
                [0.15, 1032.133, 173.513, 176.834, 3.6063, 434.764],
                [0.2, 1076.030, 231.351, 177.912, 2.2419, 400.835],
                [0.25, 1120.128, 289.188, 177.912, 2.2419, 366.996],
                [0.3, 1164.153, 347.026, 178.886, 1.5252, 333.157],
                [0.35, 1208.203, 404.864, 180.359, 0.9379, 299.317],  # 6 yehudi break
                [0.4, 1252.246, 462.701, 182.289, 0.4285, 277.288],
                [0.45, 1296.289, 520.539, 184.904, -0.2621, 263],
                [0.5, 1340.329, 578.377, 188.389, -0.6782, 248.973],
                [0.55, 1384.375, 636.214, 192.736, -0.9436, 234.816],
                [0.60, 1428.416, 694.052, 197.689, -1.2067, 220.658],
                [0.65, 1472.458, 751.890, 203.294, -1.4526, 206.501],
                [0.7, 1516.504, 809.727, 209.794, -1.6350, 192.344],
                [0.75, 1560.544, 867.565, 217.084, -1.8158, 178.186],
                [0.8, 1604.576, 925.402, 225.188, -2.0301, 164.029],
                [0.85, 1648.616, 983.240, 234.082, -2.2772, 149.872],
                [0.9, 1692.659, 1041.078, 243.625, -2.5773, 135.714],
                [0.95, 1736.710, 1098.915, 253.691, -3.1248, 121.557],
                [1.0, 1780.737, 1156.753, 263.827, -3.75, 107.4],  # 19
            ])


def getFullMesh(left_mesh=None, right_mesh=None):
    """
    For a symmetric wing, OAS only keeps and does computation on the left half.
    This script mirros the OAS mesh and attaches it to the existing mesh to
    obtain the full mesh.

    Parameters
    ----------
    left_mesh[nx,ny,3] or right_mesh : numpy array
        The half mesh to be mirrored.

    Returns
    -------
    full_mesh[nx,2*ny-1,3] : numpy array
        The computed full mesh.
    """
    if left_mesh is None and right_mesh is None:
        raise ValueError("Either the left or right mesh need to be supplied.")
    elif left_mesh is not None and right_mesh is not None:
        raise ValueError("Please only provide either left or right mesh, not both.")
    elif left_mesh is not None:
        right_mesh = np.flip(left_mesh, axis=1).copy()
        right_mesh[:, :, 1] *= -1
    else:
        left_mesh = np.flip(right_mesh, axis=1).copy()
        left_mesh[:, :, 1] *= -1
    full_mesh = np.concatenate((left_mesh, right_mesh[:, 1:, :]), axis=1)
    return full_mesh


def regen_chordwise_panels(mesh, num_x, chord_cos_spacing):
    """
    Generates a new mesh based on an existing mesh with the specified number of
    chordwise points.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface with only
        the leading and trailing edges defined.
    num_x : float
        Desired number of chordwise node points for the final mesh.
    chord_cos_spacing : float
        Blending ratio of uniform and cosine spacing in the chordwise direction.
        A value of 0. corresponds to uniform spacing and a value of 1.
        corresponds to regular cosine spacing. This increases the number of
        chordwise node points near the wingtips.

    Returns
    -------
    new_mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the final aerodynamic surface with the
        specified number of chordwise node points.

    """

    # Obtain mesh and num properties
    num_y = mesh.shape[1]

    # chordwise discretization
    cosine = 0.5 * (1 - np.cos(np.linspace(0, np.pi, num_x)))  # cosine spacing from 0 to 1
    uniform = np.linspace(0, 1, num_x)  # uniform spacing
    # mixed spacing with chord_cos_spacing as a weighting factor
    wing_x = cosine * chord_cos_spacing + (1 - chord_cos_spacing) * uniform

    # Obtain the leading and trailing edges
    le = mesh[0, :, :]
    te = mesh[-1, :, :]

    # Create a new mesh with the desired num_x and set the leading and trailing edge values
    new_mesh = np.zeros((num_x, num_y, 3))
    new_mesh[0, :, :] = le
    new_mesh[-1, :, :] = te

    for i in range(1, num_x - 1):
        w = wing_x[i]
        new_mesh[i, :, :] = (1 - w) * le + w * te

    return new_mesh



num_y = 33
num_x = 14
span_cos_spacing = 0
chord_cos_spacing = 0

# If this is a jig shape, remove all z-deflection to create a
# poor person's version of the undeformed CRM.
raw_crm_points[:, 3] = 0.0

# Get the leading edge of the raw crm points
le = np.vstack((raw_crm_points[:, 1], raw_crm_points[:, 2], raw_crm_points[:, 3]))

# Get the chord, twist(in correct order), and eta values from the points
chord = raw_crm_points[:, 5]
twist = raw_crm_points[:, 4][::-1]
eta = raw_crm_points[:, 0]


# Get the trailing edge of the crm points, based on the chord + le distance.
# Note that we do not account for twist here; instead we set that using
# the twist design variable later in run_classes.py.
te = np.vstack((raw_crm_points[:, 1] + chord, raw_crm_points[:, 2], raw_crm_points[:, 3]))

# Get the number of points that define this CRM shape and create a mesh
# array based on this size
n_raw_points = raw_crm_points.shape[0]
mesh = np.empty((2, n_raw_points, 3))

# Set the leading and trailing edges of the mesh matrix
mesh[0, :, :] = le.T
mesh[1, :, :] = te.T

# Convert the mesh points to meters from inches.
raw_mesh = mesh * 0.0254

# Index of symmetry line
ny2 = (num_y + 1) // 2


# --- spanwise discretization ---
# span_cos_spacing : float (optional)
#         Blending ratio of uniform and cosine spacing in the spanwise direction.
#         A value of 0. corresponds to uniform spacing and a value of 1.
#         corresponds to regular cosine spacing. This increases the number of
#         spanwise node points near the wingtips. A value between 2 and 3 will
#         create cosine spacing at both the root and tips.
#     chord_cos_spacing : float (optional)
#         Blending ratio of uniform and cosine spacing in the chordwise direction.
#         A value of 0. corresponds to uniform spacing and a value of 1.
#         corresponds to regular cosine spacing. This increases the number of
#         chordwise node points near the leading/trailing edge.
# Create the blended spacing using the user input for span_cos_spacing
# Spacings >= 2.0 bunch panels at both the root and tips

if span_cos_spacing >= 2.0:
    beta = np.linspace(0, np.pi, ny2)

    # mixed spacing with span_cos_spacing as a weighting factor
    # this is for the spanwise spacing
    cosine = 1 - np.cos(beta)  # cosine spacing
    uniform = np.linspace(0, 1.0, ny2)[::-1]  # uniform spacing
    lins = cosine[::-1] * (span_cos_spacing - 2.0) + (1 - (span_cos_spacing - 2.0)) * uniform
else:
    beta = np.linspace(0, np.pi / 2, ny2)

    # mixed spacing with span_cos_spacing as a weighting factor
    # this is for the spanwise spacing
    cosine = np.cos(beta)  # cosine spacing
    uniform = np.linspace(0, 1.0, ny2)[::-1]  # uniform spacing
    lins = cosine * span_cos_spacing + (1 - span_cos_spacing) * uniform


# Populate a mesh object with the desired num_y dimension based on
# interpolated values from the raw CRM points.
mesh = np.empty((2, ny2, 3))
for j in range(2):
    for i in range(3):
        mesh[j, :, i] = np.interp(lins[::-1], eta, raw_mesh[j, :, i].real)

# print('mesh 2 ', mesh)

# That is just one half of the mesh and we later expect the full mesh,
# even if we're using symmetry == True.
# So here we mirror and stack the two halves of the wing.
full_mesh = getFullMesh(right_mesh=mesh)


# If we need to add chordwise panels, do so
if num_x > 2:
    full_mesh = regen_chordwise_panels(full_mesh, num_x, chord_cos_spacing)


plt.scatter(full_mesh[:, :, 0], full_mesh[:, :, 1])
plt.axis('equal')
plt.show()


# save the mesh to a npz file
np.savez('crm_vlm_mesh_33x14.npz', mesh=full_mesh)