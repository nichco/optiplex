import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pyvista as pv

class LiftingLine:
    def __init__(self, N, b, c_root, c_tip,
                 cl_alpha=2*jnp.pi):

        self.N = N

        self.alpha = 0.0
        self.b = b
        self.S = 0.5*self.b*(c_root + c_tip)
        self.AR = self.b**2/self.S
        self.cl_alpha = cl_alpha

        self.c_root = c_root
        self.c_tip = c_tip

        delta = 0.5*jnp.pi/self.N
        self.theta = jnp.linspace(delta, jnp.pi - delta, self.N)
        self.y = 0.5*self.b*jnp.cos(self.theta)

        u = jnp.absolute(2*self.y/self.b)
        self.chord = (1.0 - u)*c_root + u*c_tip

        return

    def solve_lifting_line_model(self, x):
        A = jnp.zeros((self.N, self.N), dtype=x.dtype)
        b = jnp.zeros(self.N, dtype=x.dtype)

        for m in range(self.N):
            for n in range(self.N):
                cval = 0.25*(self.chord[m]/self.b)*self.cl_alpha
                A = A.at[m, n].set(
                    jnp.sin((n+1)*self.theta[m])*jnp.sin(self.theta[m]) +
                    cval*jnp.sin((n+1)*self.theta[m])
                )

            b = b.at[m].set(
                0.25*(self.chord[m]/self.b)*jnp.sin(self.theta[m])*self.cl_alpha*(self.alpha + x[m])
            )

        return jnp.linalg.solve(A, b)

    def compute_drag(self, coef):

        s = 0.0
        for n in range(self.N):
            s += (n+1)*coef[n]**2
        return jnp.pi*self.AR*s

    def compute_lift_coefficient(self, coef):
        return jnp.pi*self.AR*coef[0]

    def compute_lift_distribution(self, coef, rho, v_inf):

        Gamma = jnp.zeros(self.N)
        for n in range(self.N):
            Gamma = Gamma + 2.0 * self.b * coef[n] * jnp.sin((n + 1) * self.theta)

        L_prime = rho * v_inf * Gamma

        return L_prime
    
    # def compute_lift_forces(self, x, rho, v_inf):
    #     coef = self.solve_lifting_line_model(x)

    #     Gamma = jnp.zeros(self.N)
    #     for n in range(self.N):
    #         Gamma = Gamma + 2.0 * self.b * coef[n] * jnp.sin((n + 1) * self.theta)

    #     L_prime = rho * v_inf * Gamma

    #     theta_boundaries = jnp.linspace(0, jnp.pi, self.N + 1)
    #     y_boundaries = 0.5 * self.b * jnp.cos(theta_boundaries)

    #     # Panel widths — positive by construction since we take the absolute value
    #     delta_y = jnp.abs(jnp.diff(y_boundaries))

    #     F = L_prime * delta_y

    #     return F

    def plot_result(self, x):
        coef = self.solve_lifting_line_model(x)

        size = 100
        theta = jnp.linspace(0, jnp.pi, size)
        y = 0.5*self.b*jnp.cos(theta)
        Gamma = jnp.zeros(size)

        for n in range(self.N):
            Gamma = Gamma + 2*self.b*coef[n]*jnp.sin((n+1)*theta)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(self.y, x, linewidth=2)
        ax[0].set_title('Twist distribution')
        ax[1].plot(y, Gamma, linewidth=2)
        ax[1].set_title('Gamma distribution')

        return

    def plot_3d(self, x, plotter, cmap="viridis", disp=None):

        coef   = np.array(self.solve_lifting_line_model(x))
        y_span = np.array(self.y)

        Gamma = np.zeros(self.N)
        for n in range(self.N):
            Gamma += 2.0 * float(self.b) * float(coef[n]) * np.sin((n + 1) * np.array(self.theta))

        u = np.abs(2.0 * y_span / float(self.b))
        chord_span = (1.0 - u) * float(self.c_root) + u * float(self.c_tip)

        disp = np.array(disp[:, :3]) if disp is not None else np.zeros((self.N, 3))

        pts, scalars = [], []
        for j in range(self.N):
            x_le = -0.25 * chord_span[j] + disp[j, 0]
            x_te =  0.75 * chord_span[j] + disp[j, 0]
            y_j  = y_span[j]             + disp[j, 1]
            z_j  =                         disp[j, 2]
            pts.append([x_le, y_j, z_j])
            pts.append([x_te, y_j, z_j])
            scalars.extend([Gamma[j], Gamma[j]])

        pts     = np.array(pts)
        scalars = np.array(scalars)

        faces = []
        for j in range(self.N - 1):
            faces.extend([4, 2*j, 2*j+1, 2*j+3, 2*j+2])

        mesh = pv.PolyData(pts, np.array(faces))
        mesh.point_data["Gamma"] = scalars
        plotter.add_mesh(mesh, scalars="Gamma", cmap=cmap, show_edges=True)


if __name__ == "__main__":

    N = 31
    b = 15.0
    # c_root = 1.0
    # c_tip = 0.65
    c_root = 2.0
    c_tip = 0.65

    lifting_line = LiftingLine(N, b, c_root, c_tip)

    x = jnp.ones(N) * jnp.deg2rad(5)

    coef = lifting_line.solve_lifting_line_model(x)

    CD = lifting_line.compute_drag(coef)
    print('CD: ', CD)

    size = 100
    theta = jnp.linspace(0, jnp.pi, size)
    y = 0.5*b*jnp.cos(theta)
    Gamma = jnp.zeros(size)

    for n in range(N):
        Gamma = Gamma + 2*b*coef[n]*jnp.sin((n+1)*theta)

    lift_distribution = lifting_line.compute_lift_distribution(coef, rho=1.225, v_inf=100.0)
    # lift_forces = lifting_line.compute_lift_forces(coef, rho=1.225, v_inf=100.0)

    lifting_line.plot_result(x)
    plt.show()

    plotter = pv.Plotter()
    lifting_line.plot_3d(x, plotter)
    plotter.view_isometric()
    plotter.set_background("white")
    plotter.show()