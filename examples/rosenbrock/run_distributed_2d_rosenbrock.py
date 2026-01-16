from optiplex import Plex
import numpy as np
import modopt as mo
import jax.numpy as jnp
import matplotlib.pyplot as plt

v_init = [np.array([-1.0]), np.array([-1.0])]

x1_history = [v_init[0]]
x2_history = [v_init[1]]

def subproblem1(v_init, y, mu):
    x1 = v_init[0]
    x2 = v_init[1]

    v0 = x1

    def jax_obj(v):
        x1 = v[0]
        return jnp.squeeze((1 - x1)**2 + 5 * (x2 - x1**2)**2)
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf, order=1)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()
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
        return jnp.squeeze((1 - x1)**2 + 5 * (x2 - x1**2)**2)
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf, order=1)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()
    ans = optimizer.results['x']

    x1_history.append(x1)
    x2_history.append(ans)

    return [x1, ans]


opt = Plex(blocks=[subproblem1, subproblem2],
           x_init=v_init)

opt.solve(max_iter=300, tol=1e-3)


print('Solution: ', opt.solution)
print('Success: ', opt.success)
print('Iterations: ', opt.num_iter)
print('Time (s): ', opt.time)


# plt.figure(figsize=(2.5, 2.5))
plt.rcParams.update({'font.size': 14})

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 5 * (Y - X**2)**2
levels = np.linspace(0, max(Z.flatten()), 30)
plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)

plt.plot(x1_history, x2_history, '-', color='tab:red', linewidth=2.5, markersize=4, zorder=10)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)

plt.gca().set_aspect('equal')

# plt.savefig('rosenbrock_bcd.pdf', bbox_inches='tight')
plt.show()