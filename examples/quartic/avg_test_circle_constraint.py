from optiplex import Plex
import numpy as np
import modopt as mo
import jax.numpy as jnp
import matplotlib.pyplot as plt
from optiplex import combo
import warnings
warnings.filterwarnings("ignore")

# v_init = [1.0, -1.0, 1.0, -1.0]
# v_init = [-1.0, 1.0, -1.0, 1.0]
v_init = [-0.5, 1.0, -0.5, 1.0]
# v_init = [0.5, 1.0, 0.5, 1.0]

x1_1_history = [v_init[0]]
x2_1_history = [v_init[1]]
x1_2_history = [v_init[2]]
x2_2_history = [v_init[3]]


def subproblem1(v_init, y, mu):
    x1_1 = v_init[0]
    x2_1 = v_init[1]
    x1_2 = v_init[2]
    x2_2 = v_init[3]

    v0 = np.concatenate([np.atleast_1d(x1_1), np.atleast_1d(x2_1)])

    def jax_obj(v):
        x1_1 = v[0]
        x2_1 = v[1]
        beta = 1.5 # beta in [0, 2)
        obj = jnp.squeeze(x1_1**2 + x2_1**2 - beta * x1_1 * x2_1)

        c_1 = combo([x1_1, x1_2])
        c_2 = combo([x2_1, x2_2])
        c = jnp.concatenate([c_1, c_2])

        return obj + y.T @ c + mu * jnp.sum(c**2)
    
    def jax_con(v):
        x1_1 = v[0]
        x2_1 = v[1]
        con = x1_1**2 + x2_1**2
        return con.flatten()
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=0.5**2, cu=np.inf, order=1)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    ans = optimizer.results['x']

    x1_1_history.append(ans[0])
    x2_1_history.append(ans[1])
    x1_2_history.append(x1_2)
    x2_2_history.append(x2_2)

    return [ans[0], ans[1], x1_2, x2_2]


def subproblem2(v_init, y, mu):
    x1_1 = v_init[0]
    x2_1 = v_init[1]
    x1_2 = v_init[2]
    x2_2 = v_init[3]

    v0 = np.concatenate([np.atleast_1d(x1_2), np.atleast_1d(x2_2)])

    def jax_obj(v):
        x1_2 = v[0]
        x2_2 = v[1]
        beta = 1.5 # beta in [0, 2)
        obj = jnp.squeeze(x1_2**2 + x2_2**2 - beta * x1_2 * x2_2)

        # c_1 = combo([x1_1, x1_2])
        # c_2 = combo([x2_1, x2_2])
        x01_hat = jnp.average(jnp.array([x1_1, x1_2]))
        x02_hat = jnp.average(jnp.array([x2_1, x2_2]))

        c_1 = jnp.array([x1_1 - x01_hat])
        c_2 = jnp.array([x2_2 - x02_hat])
        c = jnp.concatenate([c_1, c_2])

        return obj + y.T @ c + mu * jnp.sum(c**2)
    
    def jax_con(v):
        x1_2 = v[0]
        x2_2 = v[1]
        con = x1_2**2 + x2_2**2
        return con.flatten()
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=0.5**2, cu=np.inf, order=1)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    ans = optimizer.results['x']

    x1_1_history.append(x1_1)
    x2_1_history.append(x2_1)
    x1_2_history.append(ans[0])
    x2_2_history.append(ans[1])

    return [x1_1, x2_1, ans[0], ans[1]]



# def con(v_init):
    
#     x1_1 = v_init[0]
#     x2_1 = v_init[1]
#     x1_2 = v_init[2]
#     x2_2 = v_init[3]

#     c_1 = combo([x1_1, x1_2])
#     c_2 = combo([x2_1, x2_2])
#     return jnp.concatenate([c_1, c_2])

def con(v_init):
    
    x1_1 = v_init[0]
    x2_1 = v_init[1]
    x1_2 = v_init[2]
    x2_2 = v_init[3]

    x01_hat = np.average(np.array([x1_1, x1_2]))
    x02_hat = np.average(np.array([x2_1, x2_2]))

    c_1 = np.array([x1_1 - x01_hat])
    c_2 = np.array([x2_2 - x02_hat])

    # c_1 = combo([x1_1, x1_2])
    # c_2 = combo([x2_1, x2_2])
    return np.concatenate([c_1, c_2])


opt = Plex(subproblems=[subproblem1, subproblem2],
           x_init=v_init,
           con=con,
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=10,
          eps_inner=1e-2, # inner loop tolerance
          eps_outer=1e-5, # outer loop tolerance
          tol=1e-5, # feasibility tolerance
          rho=1.1,
          mu=1.0,
          )


print('Solution: ', opt.x)
print('Time (s): ', opt.time)



plt.rcParams.update({'font.size': 14})

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 1.5 * X * Y
levels = np.linspace(0, max(Z.flatten()), 30)
plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)

plt.plot(x1_1_history, x2_2_history, 's-', color='tab:orange', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x_1, y_2)$')
plt.plot(x1_2_history, x2_1_history, 'o-', color='tab:purple', linewidth=2.5, markersize=6, zorder=10, mec='k', label=r'$(x_2, y_1)$')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

# Draw circular keep-out constraint
theta = np.linspace(0, 2*np.pi, 100)
circle_x = 0.5 * np.cos(theta)
circle_y = 0.5 * np.sin(theta)
plt.plot(circle_x, circle_y, '--', color='black', linewidth=2, alpha=0.5)
plt.fill(circle_x, circle_y, color='black', alpha=0.3)

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)

plt.legend()

plt.gca().set_aspect('equal')

# plt.savefig('augmented_lagrangian_circle_constraint.pdf', bbox_inches='tight')
plt.show()