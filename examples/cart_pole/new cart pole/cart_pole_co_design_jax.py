import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax
jax.config.update("jax_enable_x64", True)
from modopt import JaxProblem, SLSQP

# dynamics from https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
# co-design problem by me

# parameters
n = 30
dt = 2 / n
mc = 2
g = 9.81
d = 0.8
mu_cart = 0.03
mu_pole = 0.03
x0 = np.array([0, np.pi, 0, 0])
xf = np.array([d, 0, 0, 0])

def jax_obj(v):

    # v = [l, mp, x, u]
    # l: scalar, mp: scalar, x: (4, n), u: (n,)
    l = v[0]
    mp = v[1]
    x = v[2:2 + 4 * n].reshape((4, n))
    u = v[2 + 4 * n:]

    return 0.5 * dt * jnp.sum(u[:-1]**2 + u[1:]**2)


def jax_con(v):

    # v = [l, mp, x, u]
    # l: scalar, mp: scalar, x: (4, n), u: (n,)
    l = v[0]
    mp = v[1]
    x = v[2:2 + 4 * n].reshape((4, n))
    u = v[2 + 4 * n:]

    l_hat = l / 2

    theta = x[1, :]
    dx = x[2, :]
    dtheta = x[3, :]
    cart_force = u

    ddx_num = mp * g * jnp.sin(theta) * jnp.cos(theta) - (7/3) *\
          (cart_force + mp * l_hat * dtheta**2 * jnp.sin(theta) -\
            mu_cart * dx) - (mu_pole * dtheta * jnp.cos(theta) / l_hat)
    
    ddx_den = mp * jnp.cos(theta)**2 - (7/3) * (mc + mp)

    ddx = ddx_num / ddx_den

    ddtheta = 3 * (g * jnp.sin(theta) - ddx * jnp.cos(theta) - (mu_pole * dtheta / (mp * l_hat))) / (7 * l_hat)

    f = jnp.vstack((dx, dtheta, ddx, ddtheta)) # (4, n)

    # trapezoidal collocation constraints
    r = x[:, 1:] - x[:, :-1] - 0.5 * dt * (f[:, 1:] + f[:, :-1]) # (4, n-1)
    r = r.flatten()

    # initial condition constraints
    x0 = x[:, 0] # 

    # terminal constraints
    xf = x[:, n - 1]

    return jnp.concatenate((r, x0, xf))


cl_r = np.zeros((4 * (n - 1)))
cu_r = np.zeros((4 * (n - 1)))
cl = np.concatenate((cl_r, x0, xf))
cu = np.concatenate((cu_r, x0, xf))

r_scaler = np.ones(4 * (n - 1)) * 1e1
x0_scaler = np.ones(4) * 1
xf_scaler = np.ones(4) * 1
c_scaler = np.concatenate((r_scaler, x0_scaler, xf_scaler))

state_u = np.ones((4, n)) * np.inf
state_l = np.ones((4, n)) * -np.inf
l_u = np.array([5])
l_l = np.array([0.1])
mp_u = np.array([3])
mp_l = np.array([0.1])
u_u = np.ones((n)) * 50
u_l = np.ones((n)) * -50
xl = np.concatenate((l_l, mp_l, state_l.flatten(), u_l))
xu = np.concatenate((l_u, mp_u, state_u.flatten(), u_u))

l_0 = np.array([0.5])
mp_0 = np.array([0.4])
q1_0 = np.linspace(0, d, n)
q2_0 = np.linspace(np.pi, 0, n)
q3_0 = np.zeros(n)
q4_0 = np.zeros(n)
state_0 = np.vstack((q1_0, q2_0, q3_0, q4_0)).flatten()
u_0 = np.zeros(n)
v0 = np.concatenate((l_0, mp_0, state_0, u_0))

l_scaler = np.ones(1)
mp_scaler = np.ones(1)
state_scaler = np.ones(4 * n)
u_scaler = np.ones(n) * 1e-1
x_scaler = np.concatenate((l_scaler, mp_scaler, state_scaler, u_scaler))

o_scaler = 1e-2

jaxprob = JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=cl, cu=cu, 
                     xl=xl, xu=xu, o_scaler=o_scaler, c_scaler=c_scaler, x_scaler=x_scaler)
optimizer = SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-7}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()
ans = optimizer.results['x'] / x_scaler

l = ans[0]
mp = ans[1]
x = ans[2:2 + 4 * n].reshape((4, n))
u = ans[2 + 4 * n:]

print('l: ', l)
print('mp: ', mp)

position = x[0, :].flatten()
velocity = x[2, :].flatten()
angle = x[1, :].flatten()

t = np.linspace(0, n*dt, n)
plt.plot(t, angle, label='angle')
plt.plot(t, u, label='control')
plt.plot(t, position, label='position')
plt.plot(t, velocity, label='velocity')
plt.legend()
plt.show()



# Compute pole tip position
pole_x = position + l * np.sin(angle)
pole_y = l * np.cos(angle)

# Animation setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.grid(True)

# Determine plot limits
x_min, x_max = position.min() - l - 0.5, position.max() + l + 0.5
y_min, y_max = -l - 0.2, l + 0.2
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Cart parameters
cart_width = 0.4
cart_height = 0.2

# Create cart as a Rectangle patch
cart = plt.Rectangle((position[0] - cart_width/2, -cart_height/2), cart_width, cart_height, facecolor='lightblue')
ax.add_patch(cart)

# Create pole as a line
pole_line, = ax.plot([], [], lw=5, color='red')
pole_tip, = ax.plot([], [], 'o', color='green', markersize=12)

# Animation function which updates figure data.  This is called sequentially
def animate(i):
    # Update cart position
    cart.set_xy((position[i] - cart_width/2, -cart_height/2))
    # Update pole line: from cart center up to pole tip
    x0, y0 = position[i], 0
    pole_line.set_data([x0, pole_x[i]], [y0, pole_y[i]])
    pole_tip.set_data([pole_x[i]], [pole_y[i]])
    return cart, pole_line, pole_tip

# Create animation
ani = FuncAnimation(fig, animate, frames=len(t), blit=True, interval=30)

# from matplotlib.animation import PillowWriter
# ani.save("cart_pole_animation.gif", writer=PillowWriter(fps=30))

plt.show()