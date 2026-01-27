import numpy as np
import modopt as mo
import jax.numpy as jnp
import matplotlib.pyplot as plt

# https://mdobook.github.io/

rho = 1.2
mu = 1.8E-5
k = 1.2
C_L = 0.4
e = 0.8
W_0 = 1000
W_s = 8
eta_max = 0.8
v_bar = 20
sigma = 5

def jax_obj(v):
    b, c = v[0], v[1]

    S = b * c
    W = W_0 + W_s * S
    v = (2 * W / (rho * C_L * S)) ** 0.5
    q = 0.5 * rho * v ** 2
    S_wet = 2.05 * S
    Re = rho * v * c / mu
    C_f = 0.074 / Re ** 0.2
    viscous_drag = k * C_f * q * S_wet
    induced_drag = W**2 / (q * np.pi * b**2 * e)
    total_drag = viscous_drag + induced_drag
    eta = eta_max * jnp.exp(-(v - v_bar)**2 / (2 * sigma**2))
    power_required = total_drag * v / eta
    
    return power_required
    
x0 = np.array([15.0, 1.2])

xl = np.array([0.0, 1e-2])
xu = np.array([np.inf, np.inf])

jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=xl, xu=xu)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 100, 'ftol': 1e-7}, turn_off_outputs=True)
optimizer.solve()

optimizer.print_results()
ans = optimizer.results['x']

print('wing span: ', ans[0])
print('chord: ', ans[1])





b_vals = np.linspace(5, 35, 100)
c_vals = np.linspace(0.3, 1.5, 100)
B, C = np.meshgrid(b_vals, c_vals)

S = B * C
W = W_0 + W_s * S
v = np.sqrt(2 * W / (rho * C_L * S))
q = 0.5 * rho * v**2
S_wet = 2.05 * S
Re = rho * v * C / mu
C_f = 0.074 / Re**0.2

viscous_drag = k * C_f * q * S_wet
induced_drag = W**2 / (q * np.pi * B**2 * e)
total_drag = viscous_drag + induced_drag

eta = eta_max * np.exp(-(v - v_bar)**2 / (2 * sigma**2))
Z = total_drag * v / eta

levels = np.linspace(min(Z.flatten()), 3000, 30)
plt.contour(B, C, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
plt.contourf(B, C, Z, levels=levels, cmap='Blues_r', alpha=0.5)

plt.plot(ans[0], ans[1], marker='o')
plt.xlabel('Wing span')
plt.ylabel('Chord')

plt.show()