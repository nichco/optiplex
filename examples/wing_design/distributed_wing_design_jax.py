from optiplex import Plex
import numpy as np
import modopt as mo
import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


rho = 1.2
visc = 1.8E-5
k = 1.2
C_L = 0.4
e = 0.8
W_0 = 1000
W_s = 8
eta_max = 0.8
v_bar = 20
sigma = 5


v_init = [np.array([15.0]), np.array([1.2])]

b_history = [v_init[0]]
c_history = [v_init[1]]

def subproblem1(v_init, y, mu):
    b = v_init[0]
    c = v_init[1]

    v0 = b

    def jax_obj(v):
        b = v[0]

        S = b * c
        W = W_0 + W_s * S
        v = (2 * W / (rho * C_L * S)) ** 0.5
        q = 0.5 * rho * v ** 2
        S_wet = 2.05 * S
        Re = rho * v * c / visc
        C_f = 0.074 / Re ** 0.2
        viscous_drag = k * C_f * q * S_wet
        induced_drag = W**2 / (q * np.pi * b**2 * e)
        total_drag = viscous_drag + induced_drag
        eta = eta_max * jnp.exp(-(v - v_bar)**2 / (2 * sigma**2))
        power_required = total_drag * v / eta

        return jnp.squeeze(power_required)
    
    xl = np.array([0.0])
    xu = np.array([np.inf])
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=xl, xu=xu)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-10}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()
    ans = optimizer.results['x']

    b_history.append(ans)
    c_history.append(c)

    return [ans, c]

def subproblem2(v_init, y, mu):
    b = v_init[0]
    c = v_init[1]

    v0 = c

    def jax_obj(v):
        c = v[0]

        S = b * c
        W = W_0 + W_s * S
        v = (2 * W / (rho * C_L * S)) ** 0.5
        q = 0.5 * rho * v ** 2
        S_wet = 2.05 * S
        Re = rho * v * c / visc
        C_f = 0.074 / Re ** 0.2
        viscous_drag = k * C_f * q * S_wet
        induced_drag = W**2 / (q * np.pi * b**2 * e)
        total_drag = viscous_drag + induced_drag
        eta = eta_max * jnp.exp(-(v - v_bar)**2 / (2 * sigma**2))
        power_required = total_drag * v / eta

        return jnp.squeeze(power_required)
    
    xl = np.array([1e-2])
    xu = np.array([np.inf])
    
    jaxprob = mo.JaxProblem(x0=v0, jax_obj=jax_obj, xl=xl, xu=xu)

    optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 300, 'ftol': 1e-7}, turn_off_outputs=True)
    optimizer.solve()
    optimizer.print_results()
    ans = optimizer.results['x']

    b_history.append(b)
    c_history.append(ans)

    return [b, ans]


opt = Plex(blocks=[subproblem1, subproblem2],
           x_init=v_init)

opt.solve(max_iter=300, tol=1e-5, itol=1)


print('Solution: ', opt.x)
print('Success: ', opt.success)
print('Iterations: ', opt.dual_iterations)
print('Time (s): ', opt.time)


plt.figure(figsize=(4, 4))
# plt.rcParams.update({'font.size': 14})

b_vals = np.linspace(5, 35, 100)
c_vals = np.linspace(0.3, 1.5, 100)
B, C = np.meshgrid(b_vals, c_vals)

S = B * C
W = W_0 + W_s * S
v = np.sqrt(2 * W / (rho * C_L * S))
q = 0.5 * rho * v**2
S_wet = 2.05 * S
Re = rho * v * C / visc
C_f = 0.074 / Re**0.2

viscous_drag = k * C_f * q * S_wet
induced_drag = W**2 / (q * np.pi * B**2 * e)
total_drag = viscous_drag + induced_drag

eta = eta_max * np.exp(-(v - v_bar)**2 / (2 * sigma**2))
Z = total_drag * v / eta

levels = np.linspace(min(Z.flatten()), 3000, 30)
plt.contour(B, C, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(B, C, Z, levels=levels, cmap='Blues_r', alpha=0.5)

dZ_dc, dZ_db = np.gradient(Z, c_vals, b_vals)

# Zero level sets of the derivatives
plt.contour(B, C, dZ_db, levels=[0], colors='tab:purple', linewidths=2, linestyles='-.', alpha=1)
plt.contour(B, C, dZ_dc, levels=[0], colors='tab:olive', linewidths=2, linestyles='-.', alpha=1)

plt.plot(b_history, c_history, '-o', mec='k', color='tab:red', linewidth=2.5, markersize=7, zorder=10)
plt.xlim(5, 35)
plt.ylim(0.3, 1.5)
plt.xlabel('Wing span')
plt.ylabel('Chord')

plt.savefig('wing_design.pdf', bbox_inches='tight')
plt.show()