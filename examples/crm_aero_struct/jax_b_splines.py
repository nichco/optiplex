import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
 
def get_bspline_mtx(num_cp, num_pt, order=4):
    order = min(order, num_cp)
    knots = np.zeros(num_cp + order)
    knots[order - 1:num_cp + 1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp + 1:] = 1.0
    t_vec = np.linspace(0, 1, num_pt)
    basis = np.zeros(order)
    arange = np.arange(order)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)
    for ipt in range(num_pt):
        t = t_vec[ipt]
        i0 = -1
        for ind in range(order, num_cp + 1):
            if (knots[ind - 1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order
        basis[:] = 0.
        basis[-1] = 1.
        for i in range(2, order + 1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n + l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1 - 1] = 0.
            for j in range(j1 + 1, j2):
                n = i0 + j
                if knots[n + l - 1] != knots[n - 1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j - 1] = 0.
                if knots[n + l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n + l - 1] != knots[n - 1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2 - 1] = 0.
        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange
    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()
    return scipy.sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_pt, num_cp),
    )
 
 
# class BsplineComp():
#     """
#     Translates control points to actual points using a B-spline.
#     """
#     def __init__(self, num_cp, num_pt, jac):
#         super().__init__()
#         self.num_cp = num_cp
#         self.num_pt = num_pt
#         self.jac = jac
 
#     def evaluate(self, ycp):

#         return self.jac @ ycp



if __name__ == "__main__":

    num_cp = 11
    num_interp = 301
    jac = get_bspline_mtx(num_cp, num_interp)
    jac_dense = jnp.asarray(jac.toarray())

    y_cp = np.linspace(0, 1, num_cp)

    def y_interp_fn(ycp):
        return jac_dense @ ycp

    y_cp_jax = jnp.asarray(y_cp)
    y_interp = np.asarray(y_interp_fn(y_cp_jax))
    y_interp_jac = np.asarray(jax.jacobian(y_interp_fn)(y_cp_jax))

    plt.plot(y_interp)
    plt.scatter(np.linspace(0, num_interp, num_cp), y_cp)
    plt.show()

    print(y_interp_jac)