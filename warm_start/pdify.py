import numpy as np




def eig_clip(H, eps=1e-6):

    H = (H + H.T) / 2 # force symmetry

    e, v = np.linalg.eigh(H)
    
    e = np.clip(e, a_min=eps, a_max=None)

    H = v.dot(np.diag(e)).dot(v.T)

    H = (H + H.T) / 2 # force symmetry
    
    return H


def eig_flip(H):

    H = (H + H.T) / 2 # force symmetry

    e, v = np.linalg.eigh(H)

    e = np.abs(e)

    H = v.dot(np.diag(e)).dot(v.T)

    H = (H + H.T) / 2 # force symmetry

    return H


def add_diag_a(H, eps=1e-6):
    # https://math.stackexchange.com/questions/332456/how-to-make-a-matrix-positive-semidefinite

    diag = np.diag(H)

    H_minus_diag = H - np.diag(H)

    row_sums = np.sum(np.abs(H_minus_diag), axis=0)

    new_diag = diag + row_sums + eps

    H = H_minus_diag + np.diag(new_diag)

    H = (H + H.T) / 2 # force symmetry

    return H







# h = np.array(
#         [
#             [0.1, 1.0, -2.0],
#             [1.0, 0.2, 1.5],
#             [-2.0, 1.5, 0.3],
#         ]
#     )

A = np.array([[0, 1, 2, 3, 4],
              [1, 0, 1, 2, 3],
              [2, 1, 0, 1, 2],
              [3, 2, 1, 0, 1],
              [4, 3, 2, 1, 0]])



h_clipped = eig_clip(A, eps=1e-6)
print(np.linalg.eigvals(h_clipped))
print("\nClipped Hessian:\n", h_clipped)

h_flipped = eig_flip(A)
print(np.linalg.eigvals(h_flipped))
print("\nFlipped Hessian:\n", h_flipped)

h_diag_a = add_diag_a(A, eps=1e-6)
print(np.linalg.eigvals(h_diag_a))
print("\nDiagonal Perturbed Hessian:\n", h_diag_a)