import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv


class LiftingLine:
    def __init__(self, le_points, te_points, v_inf, rho, cl_alpha=2 * jnp.pi):
        """
        Parameters
        ----------
        le_points : np.ndarray, shape (M, 3)
            Leading-edge points, one row per spanwise station.
            Stations must be ordered from one wingtip to the other
            (i.e. monotonically increasing or decreasing in y).
        te_points : np.ndarray, shape (M, 3)
            Trailing-edge points corresponding to each LE station.
        v_inf     : float   Free-stream velocity magnitude.
        rho       : float   Air density.
        cl_alpha  : float   Lift-curve slope (default 2π).
        """
        le_points = np.asarray(le_points, dtype=float)
        te_points = np.asarray(te_points, dtype=float)

        if le_points.shape != te_points.shape:
            raise ValueError("le_points and te_points must have the same shape.")
        if le_points.ndim != 2 or le_points.shape[1] != 3:
            raise ValueError("le_points and te_points must be (M, 3) arrays.")

        self.N = le_points.shape[0] # number of stations

        # Spanwise coordinate at each station (y column)
        y_mesh = le_points[:, 1]

        # Sort stations by y so that theta increases monotonically
        order = np.argsort(y_mesh)
        le_sorted = le_points[order]
        te_sorted = te_points[order]

        y_sorted = le_sorted[:, 1] # shape (N,)

        # Semi-span: half the full tip-to-tip extent
        self.b = float(y_sorted[-1] - y_sorted[0])   # full span

        # Chord at each station
        chord_mesh = np.linalg.norm(te_sorted - le_sorted, axis=1)  # shape (N,)

        # Planform area via trapezoidal integration over the sorted stations
        self.S = float(np.trapz(chord_mesh, y_sorted))
        self.AR = self.b ** 2 / self.S

        self.cl_alpha = cl_alpha
        self.v_inf = v_inf
        self.rho = rho

        #  Cosine-spaced collocation angles θ in (0, pi), avoiding endpoints
        delta = 0.5 * jnp.pi / self.N
        self.theta = jnp.linspace(delta, jnp.pi - delta, self.N)

        # y positions of collocation points in the cosine spacing
        # y = (b/2) cos θ  →  ranges from ~+b/2 to ~-b/2
        y_half = 0.5 * self.b
        self.y = y_half * jnp.cos(self.theta)          # shape (N,)

        #  Interpolate chord from the mesh onto the cosine-spaced stations
        # np.interp requires x to be increasing; y_sorted is already sorted
        self.chord = jnp.array(np.interp(np.array(self.y), y_sorted, chord_mesh))

        #  Store LE x-position interpolated onto cosine stations (for plot_3d)
        le_x_mesh = le_sorted[:, 0]
        self._le_x = np.interp(np.array(self.y), y_sorted, le_x_mesh)

        self.alpha = 0.0

    def solve_lifting_line_model(self, x):
        """
        x : jnp.ndarray, shape (N,)
            Geometric twist angle (radians) at each collocation station.

        Returns
        -------
        coef : jnp.ndarray, shape (N,)
            Fourier coefficients of the spanwise circulation.
        """
        A = jnp.zeros((self.N, self.N), dtype=x.dtype)
        b = jnp.zeros(self.N, dtype=x.dtype)

        for m in range(self.N):
            for n in range(self.N):
                cval = 0.25 * (self.chord[m] / self.b) * self.cl_alpha
                A = A.at[m, n].set(
                    jnp.sin((n + 1) * self.theta[m]) * jnp.sin(self.theta[m])
                    + cval * jnp.sin((n + 1) * self.theta[m])
                )
            b = b.at[m].set(
                0.25
                * (self.chord[m] / self.b)
                * jnp.sin(self.theta[m])
                * self.cl_alpha
                * (self.alpha + x[m])
            )

        return jnp.linalg.solve(A, b)

    def compute_drag(self, coef):
        s = 0.0
        for n in range(self.N):
            s += (n + 1) * coef[n] ** 2
        return jnp.pi * self.AR * s

    def compute_lift_coefficient(self, coef):
        return jnp.pi * self.AR * coef[0]

    def circulation(self, coef):
        Gamma = jnp.zeros(self.N)
        for n in range(self.N):
            Gamma = (
                Gamma
                + 2.0 * self.b * self.v_inf * coef[n] * jnp.sin((n + 1) * self.theta)
            )
        return Gamma

    def compute_forces(self, coef):
        """
        Returns panel forces, shape (N, 3) [F_x, F_y, F_z].
            F_x : induced drag (streamwise, positive downstream)
            F_y : zero (by symmetry)
            F_z : lift (normal)
        """
        Gamma = self.circulation(coef)

        alpha_i = jnp.zeros(self.N)
        for n in range(self.N):
            alpha_i = (
                alpha_i
                + (n + 1)
                * coef[n]
                * jnp.sin((n + 1) * self.theta)
                / jnp.sin(self.theta)
            )

        w_i = -self.v_inf * alpha_i

        theta_bnd = jnp.linspace(0.0, jnp.pi, self.N + 1)
        y_bnd = 0.5 * self.b * jnp.cos(theta_bnd)
        delta_y = jnp.abs(jnp.diff(y_bnd))

        F_x = -self.rho * Gamma * w_i * delta_y
        F_y = jnp.zeros(self.N)
        F_z = self.rho * self.v_inf * Gamma * delta_y

        return jnp.stack([F_x, F_y, F_z], axis=1) # (N, 3)

    def plot_3d(self, x, plotter, cmap="viridis", disp=None):
        coef = np.array(self.solve_lifting_line_model(x))
        y_span = np.array(self.y)
        chord_span = np.array(self.chord)
        le_x = self._le_x # x-coordinate of the LE at each station

        Gamma = self.circulation(coef)

        disp = np.array(disp[:, :3]) if disp is not None else np.zeros((self.N, 3))

        pts, scalars, centers = [], [], []
        for j in range(self.N):
            x_le = le_x[j]                       + disp[j, 0]
            x_te = le_x[j] + chord_span[j]       + disp[j, 0]
            y_j  = y_span[j]                     + disp[j, 1]
            z_j  =                                 disp[j, 2]
            pts.append([x_le, y_j, z_j])
            pts.append([x_te, y_j, z_j])
            scalars.extend([Gamma[j], Gamma[j]])
            centers.append(np.array([0.5 * (x_le + x_te), y_j, z_j]))

        pts = np.array(pts)
        scalars = np.array(scalars)

        faces = []
        for j in range(self.N - 1):
            faces.extend([4, 2 * j, 2 * j + 1, 2 * j + 3, 2 * j + 2])

        mesh = pv.PolyData(pts, np.array(faces))
        mesh.point_data["Gamma"] = scalars
        plotter.add_mesh(mesh, scalars="Gamma", cmap=cmap, show_edges=True)

        forces = np.array(self.compute_forces(coef))
        mag = np.linalg.norm(forces, axis=1)

        cloud = pv.PolyData(np.array(centers))
        cloud["forces"] = forces

        glyphs = cloud.glyph(
            orient="forces",
            scale="forces",
            factor=0.2 * self.b / (np.max(mag) + 1e-12),
        )
        plotter.add_mesh(glyphs, color="red")


#  Helper: build LE/TE arrays from legacy scalar parameters
def build_planform_mesh(N, b, c_root, c_tip):
    """
    Reconstruct the (N, 3) LE and TE arrays that a simple tapered wing would
    produce, using the same cosine spacing as LiftingLine.

    Returns
    -------
    le_points : np.ndarray, shape (N, 3)
    te_points : np.ndarray, shape (N, 3)
    """
    delta = 0.5 * np.pi / N
    theta = np.linspace(delta, np.pi - delta, N)
    y = 0.5 * b * np.cos(theta) # tip-to-tip, cosine spaced

    u = np.abs(2.0 * y / b)
    chord = (1.0 - u) * c_root + u * c_tip

    # LE at x = -c/4 (quarter-chord convention), TE at x = +3c/4
    le_points = np.stack([-0.25 * chord, y, np.zeros(N)], axis=1)
    te_points = np.stack([ 0.75 * chord, y, np.zeros(N)], axis=1)

    return le_points, te_points




if __name__ == "__main__":

    N      = 31
    b      = 15.0
    c_root = 1.0
    c_tip  = 0.65
    v_inf  = 100.0
    rho    = 1.225

    # Build mesh from the legacy parameters and pass it in
    le_points, te_points = build_planform_mesh(N, b, c_root, c_tip)

    lifting_line = LiftingLine(le_points, te_points, v_inf, rho)

    x = jnp.ones(lifting_line.N) * jnp.deg2rad(5)

    coef = lifting_line.solve_lifting_line_model(x)

    CD = lifting_line.compute_drag(coef)
    print("CD:", CD)

    Gamma = lifting_line.circulation(coef)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(lifting_line.y, x, linewidth=2)
    ax[0].set_title("Twist")
    ax[1].plot(lifting_line.y, Gamma, linewidth=2)
    ax[1].set_title("Gamma")
    plt.show()

    plotter = pv.Plotter()
    lifting_line.plot_3d(x, plotter)
    plotter.view_isometric()
    plotter.show()

    forces = lifting_line.compute_forces(coef)
    print("Forces (F_x, F_y, F_z) at each panel:\n", forces)

    q_inf      = 0.5 * rho * v_inf ** 2
    CL_series  = lifting_line.compute_lift_coefficient(coef)
    CL_panels  = jnp.sum(forces[:, 2]) / (q_inf * lifting_line.S)
    CDi_series = lifting_line.compute_drag(coef)
    CDi_panels = jnp.sum(forces[:, 0]) / (q_inf * lifting_line.S)

    print(f"CL  — series: {CL_series:.4f}  |  panels: {CL_panels:.4f}")
    print(f"CDi — series: {CDi_series:.4f}  |  panels: {CDi_panels:.4f}")