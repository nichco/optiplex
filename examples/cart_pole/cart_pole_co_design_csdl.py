import csdl_alpha as csdl
import numpy as np
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# dynamics from https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
# co-design problem by me

n = 30
dt = 2 / n
mc = 2
g = 9.81
d = 0.8
mu_cart = 0.03
mu_pole = 0.03



recorder = csdl.Recorder(inline=True)
recorder.start()


l = csdl.Variable(value=0.5)
l.set_as_design_variable(lower=0.1, upper=5, scaler=1)
l_hat = l / 2

mp = csdl.Variable(value=0.4)
mp.set_as_design_variable(lower=0.1, upper=3, scaler=1)


q1_0 = np.linspace(0, d, n)
q2_0 = np.linspace(np.pi, 0, n)
q3_0 = np.zeros(n)
q4_0 = np.zeros(n)
x = csdl.Variable(value=np.vstack((q1_0, q2_0, q3_0, q4_0)))
x.set_as_design_variable(scaler=1)

u = csdl.Variable(value=np.zeros((n)))
u.set_as_design_variable(lower=-50, upper=50, scaler=1E-1)

# dynamics
f = csdl.Variable(value=np.zeros((4, n)))
for i in csdl.frange(n):

    cart_force = u[i]
    theta = x[1, :][i]
    dx = x[2, :][i]
    dtheta = x[3, :][i]

    ddx_num = mp * g * csdl.sin(theta) * csdl.cos(theta) - (7/3) * (cart_force + mp * l_hat * dtheta**2 * csdl.sin(theta) - mu_cart * dx) - (mu_pole * dtheta * csdl.cos(theta) / l_hat)
    ddx_den = mp * csdl.cos(theta)**2 - (7/3) * (mc + mp)
    ddx = ddx_num / ddx_den

    ddtheta = 3 * (g * csdl.sin(theta) - ddx * csdl.cos(theta) - (mu_pole * dtheta / (mp * l_hat))) / (7 * l_hat)

    f = f.set(csdl.slice[0, i], x[2, i])
    f = f.set(csdl.slice[1, i], x[3, i])
    f = f.set(csdl.slice[2, i], ddx)
    f = f.set(csdl.slice[3, i], ddtheta)

# trapezoidal collocation constraints
r = x[:, 1:] - x[:, :-1] - 0.5 * dt * (f[:, 1:] + f[:, :-1])

r.set_as_constraint(equals=0, scaler=1E1)

# initial condition constraints
x[:, 0].set_as_constraint(equals=np.array([0, np.pi, 0, 0]), scaler=1)

# terminal constraints
x[:, n - 1].set_as_constraint(equals=np.array([d, 0, 0, 0]), scaler=1)

j = 0.5 * dt * csdl.sum(u[:-1]**2 + u[1:]**2)
j.set_as_objective(scaler=1E-2)

recorder.stop()


sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(simulator=sim)
optimizer = SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1e-7}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()


print('objective: ', j.value)
print('mp: ', mp.value)
print('length: ', l.value)


x = x.value
u = u.value
l = l.value
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