from optiplex import Plex
import numpy as np
import modopt as mo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# v_init = [np.array([1.0]), np.array([-1.0])]
v_init = [np.array([1.0]), np.array([0.5])]

x1_history = [v_init[0]]
x2_history = [v_init[1]]

def subproblem1(v_init, y, mu):
    x1 = v_init[0]
    x2 = v_init[1]

    v0 = x1

    def jax_obj(v):
        x1 = v[0]
        beta = 1.5 # beta in [0, 2)
        return jnp.squeeze(x1**2 + x2**2 - beta * x1 * x2)
    
    def jax_con(v):
        x1 = v[0]
        return x1 - 0.5*x2
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=0., cu=np.inf, order=1)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    ans = optimizer.results['x']

    x1_history.append(ans)
    x2_history.append(x2)

    return [ans, x2]

def subproblem2(v_init, y, mu):
    x1 = v_init[0]
    x2 = v_init[1]

    v0 = x2

    def jax_obj(v):
        x2 = v[0]
        beta = 1.5 # beta in [0, 2)
        return jnp.squeeze(x1**2 + x2**2 - beta * x1 * x2)
    
    def jax_con(v):
        x2 = v[0]
        return x1 - 0.5*x2
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=0, cu=np.inf, order=1)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    # optimizer.print_results()
    ans = optimizer.results['x']

    x1_history.append(x1)
    x2_history.append(ans)

    return [x1, ans]


opt = Plex(blocks=[subproblem1, subproblem2],
           x_init=v_init)

opt.solve(max_iter=100, tol=1e-5)


print('Solution: ', opt.x)
print('Success: ', opt.success)
print('Iterations: ', opt.dual_iterations)
print('Time (s): ', opt.time)


# plt.figure(figsize=(2.5, 2.5))
plt.rcParams.update({'font.size': 14})

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 1.5 * X * Y
levels = np.linspace(0, max(Z.flatten()), 30)
plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)

plt.plot(x1_history, x2_history, 'o-', color='tab:red', linewidth=2.5, markersize=6, mec='k', zorder=10)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

# constraint = x + 0.5 * y
plt.plot(x, 2*x, '--', color='k', linewidth=2, alpha=0.5)

plt.fill_between(
    x,
    1.5,        # top of plot
    2*x,          # constraint line
    color='k',
    alpha=0.4,
)

# Partial derivatives
dZ_dx1 = 2 * X - 1.5 * Y
dZ_dx2 = 2 * Y - 1.5 * X

# Zero level sets of the derivatives
plt.contour(X, Y, dZ_dx1, levels=[0], colors='tab:purple', linewidths=2, linestyles='dotted', alpha=1)
plt.contour(X, Y, dZ_dx2, levels=[0], colors='tab:olive', linewidths=2, linestyles='dotted', alpha=1)

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)

plt.gca().set_aspect('equal')

# plt.savefig('global_constraint_example.pdf', bbox_inches='tight')
plt.show()