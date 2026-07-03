from optiplex import Plex
import numpy as np
import modopt as mo
import jax.numpy as jnp
import pyvista as pv
import warnings
warnings.filterwarnings("ignore")


eps = 0.01

v_init = [
    np.array([-1.0 - eps]),
    np.array([ 1.0 + 0.5 * eps]),
    np.array([-1.0 - 0.25 * eps])
]


x1_history = [v_init[0]]
x2_history = [v_init[1]]
x3_history = [v_init[2]]
f_history = []

def unstable(x1, x2, x3):
    return jnp.squeeze(
            -x1*x2 - x2*x3 - x1*x3 + 
            jnp.maximum(x1 - 1, 0)**2 + 
            jnp.maximum(-x1 - 1, 0)**2 + 
            jnp.maximum(x2 - 1, 0)**2 + 
            jnp.maximum(-x2 - 1, 0)**2 + 
            jnp.maximum(x3 - 1, 0)**2 + 
            jnp.maximum(-x3 - 1, 0)**2
        )


def subproblem1(v_init, y, mu):

    x1 = v_init[0]
    x2 = v_init[1]
    x3 = v_init[2]

    v0 = x1

    def jax_obj(v):

        x1 = v

        return unstable(x1, x2, x3)

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=-2, xu=2)
    optimizer = mo.SLSQP(jaxprob, solver_options={"maxiter":1000, "ftol":1e-16}, turn_off_outputs=True)
    optimizer.solve()

    ans = optimizer.results["x"]
    f = optimizer.results["fun"]

    x1_history.append(ans)
    x2_history.append(x2)
    x3_history.append(x3)
    f_history.append(f)

    return [ans, x2, x3]

def subproblem2(v_init, y, mu):

    x1 = v_init[0]
    x2 = v_init[1]
    x3 = v_init[2]

    v0 = x2

    def jax_obj(v):

        x2 = v

        return unstable(x1, x2, x3)

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=-2, xu=2)
    optimizer = mo.SLSQP(jaxprob, solver_options={"maxiter":1000, "ftol":1e-16}, turn_off_outputs=True)
    optimizer.solve()

    ans = optimizer.results["x"]
    f = optimizer.results["fun"]

    x1_history.append(x1)
    x2_history.append(ans)
    x3_history.append(x3)
    f_history.append(f)

    return [x1, ans, x3]


def subproblem3(v_init, y, mu):

    x1 = v_init[0]
    x2 = v_init[1]
    x3 = v_init[2]

    v0 = x3

    def jax_obj(v):

        x3 = v

        return unstable(x1, x2, x3)

    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=-2, xu=2)
    optimizer = mo.SLSQP(jaxprob, solver_options={"maxiter":1000, "ftol":1e-16}, turn_off_outputs=True)
    optimizer.solve()

    ans = optimizer.results["x"]
    f = optimizer.results["fun"]

    x1_history.append(x1)
    x2_history.append(x2)
    x3_history.append(ans)
    f_history.append(f)

    return [x1, x2, ans]


opt = Plex(
    subproblems=[
        subproblem1,
        subproblem2,
        subproblem3,
    ],
    x_init=v_init
)

opt.solve(
    max_inner_iter=6,
    eps_inner=1e-15
)

print("Solution:", opt.x)

print("\nIterates")
for k in range(len(x1_history)):
    print(
        k,
        float(x1_history[k]),
        float(x2_history[k]),
        float(x3_history[k]),
    )

print(f_history)


xs = np.array(x1_history, dtype=float).flatten()
ys = np.array(x2_history, dtype=float).flatten()
zs = np.array(x3_history, dtype=float).flatten()

points = np.column_stack([xs, ys, zs])

# Function (NumPy version)
def unstable_np(x1, x2, x3):
    return (
        -x1*x2 - x2*x3 - x1*x3
        + np.maximum(x1 - 1, 0)**2
        + np.maximum(-x1 - 1, 0)**2
        + np.maximum(x2 - 1, 0)**2
        + np.maximum(-x2 - 1, 0)**2
        + np.maximum(x3 - 1, 0)**2
        + np.maximum(-x3 - 1, 0)**2
    )

cube = pv.Cube(bounds=(-1, 1, -1, 1, -1, 1))

def eval_points(x, y, z):
    return unstable_np(x, y, z)

cube["F"] = eval_points(cube.points[:, 0],
                        cube.points[:, 1],
                        cube.points[:, 2])

plotter = pv.Plotter()

plotter.add_mesh(
    cube,
    scalars="F",
    cmap="viridis",
    opacity=0.6,
    show_edges=True,
    smooth_shading=False,
    show_scalar_bar=False
)

trajectory = pv.lines_from_points(points)

plotter.add_mesh(
    trajectory,
    color="red",
    line_width=5,
)

plotter.add_points(
    points,
    color="red",
    point_size=15
)

plotter.show()