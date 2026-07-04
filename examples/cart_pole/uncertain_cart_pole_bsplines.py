import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import jax
jax.config.update("jax_enable_x64", True)
from modopt import JaxProblem, SLSQP
from scipy.stats.qmc import LatinHypercube, scale
from jax_b_splines import get_bspline_mtx, bspline_comp

# sample the space of uncertain parameters using Latin hypercube sampling
# uncertain parameters: g, mu_cart, mu_pole
g_u = 9.81 + 0.001
g_l = 9.81 - 0.001
mu_cart_u = 0.03 + 0.0001
mu_cart_l = 0.03 - 0.0001
mu_pole_u = 0.03 + 0.0001
mu_pole_l = 0.03 - 0.0001
N = 2
# sampler = LatinHypercube(d=3, seed=0)
# samples = scale(sampler.random(N), 
#                 l_bounds=[g_l, mu_cart_l, mu_pole_l], 
#                 u_bounds=[g_u, mu_cart_u, mu_pole_u])
samples = np.array([[9.81, 0.03, 0.03], [9.81, 0.03, 0.03]])
print(samples)

# parameters
n = 30
dt = 2 / n
mc = 2
d = 0.8
x0 = np.array([0, np.pi, 0, 0])
xf = np.array([d, 0, 0, 0])
nu = 12
bspline_mtx = get_bspline_mtx(nu, n)

ni = 0
ni += 4 * n # num state vars
ni += nu # num control vars
# l and mp are the final two vars

def jax_obj(v):

    j = 0
    for i in range(N):
        v_i = v[i * ni:(i + 1) * ni]
        u_cp_i = v_i[4 * n:]
        u_i = bspline_comp(bspline_mtx, u_cp_i)
        j_i = 0.5 * dt * jnp.sum(u_i[:-1]**2 + u_i[1:]**2)
        j += j_i
    # j = jnp.array(j_i)

    return j / N


def jax_con(v):

    l = v[-2]
    mp = v[-1]
    l_hat = l / 2

    con = []

    for i in range(N):
        v_i = v[i * ni:(i + 1) * ni]

        g_i = samples[i, 0]
        mu_cart_i = samples[i, 1]
        mu_pole_i = samples[i, 2]

        x_i = v_i[:4 * n].reshape((4, n))
        u_cp_i = v_i[4 * n:]
        u_i = bspline_comp(bspline_mtx, u_cp_i)

        theta_i = x_i[1, :]
        dx_i = x_i[2, :]
        dtheta_i = x_i[3, :]
        cart_force_i = u_i

        ddx_num_i = mp * g_i * jnp.sin(theta_i) * jnp.cos(theta_i) - (7/3) *\
            (cart_force_i + mp * l_hat * dtheta_i**2 * jnp.sin(theta_i) -\
                mu_cart_i * dx_i) - (mu_pole_i * dtheta_i * jnp.cos(theta_i) / l_hat)
        
        ddx_den_i = mp * jnp.cos(theta_i)**2 - (7/3) * (mc + mp)

        ddx_i = ddx_num_i / ddx_den_i

        ddtheta_i = 3 * (g_i * jnp.sin(theta_i) - ddx_i * jnp.cos(theta_i) - (mu_pole_i * dtheta_i / (mp * l_hat))) / (7 * l_hat)

        f_i = jnp.vstack((dx_i, dtheta_i, ddx_i, ddtheta_i)) # (4, n)

        # trapezoidal collocation constraints
        r_i = x_i[:, 1:] - x_i[:, :-1] - 0.5 * dt * (f_i[:, 1:] + f_i[:, :-1]) # (4, n-1)
        r_i = r_i.flatten()

        # initial condition constraints
        x0_i = x_i[:, 0] # 

        # terminal constraints
        xf_i = x_i[:, n - 1]

        con_i = jnp.concatenate((r_i, x0_i, xf_i))
        con.append(con_i)

    return jnp.concatenate(con)





cl_r = np.zeros((4 * (n - 1)))
cu_r = np.zeros((4 * (n - 1)))
cl_i = np.concatenate((cl_r, x0, xf))
cu_i = np.concatenate((cu_r, x0, xf))
cl = np.tile(cl_i, N)
cu = np.tile(cu_i, N)

r_scaler = np.ones(4 * (n - 1)) * 1e1
x0_scaler = np.ones(4) * 1
xf_scaler = np.ones(4) * 1
c_scaler_i = np.concatenate((r_scaler, x0_scaler, xf_scaler))
c_scaler = np.tile(c_scaler_i, N)

position_l = np.ones((n)) * 0#-np.inf
position_u = np.ones((n)) * 2#np.inf
theta_l = np.ones((n)) * -np.inf
theta_u = np.ones((n)) * np.inf
dx_l = np.ones((n)) * -4#-np.inf
dx_u = np.ones((n)) * 4#np.inf
dtheta_l = np.ones((n)) * -np.inf
dtheta_u = np.ones((n)) * np.inf
state_l = np.vstack((position_l, theta_l, dx_l, dtheta_l))
state_u = np.vstack((position_u, theta_u, dx_u, dtheta_u))
# state_u = np.ones((4, n)) * np.inf
# state_l = np.ones((4, n)) * -np.inf
l_u = np.array([5])
l_l = np.array([0.1])
mp_u = np.array([3])
mp_l = np.array([0.1])
u_u = np.ones((nu)) * 50
u_l = np.ones((nu)) * -50
# u_u[0] = 0
# u_l[0] = 0
# u_u[-1] = 0
# u_l[-1] = 0
xl_i = np.concatenate((state_l.flatten(), u_l))
xu_i = np.concatenate((state_u.flatten(), u_u))
xl = np.concatenate((np.tile(xl_i, N), l_l, mp_l))
xu = np.concatenate((np.tile(xu_i, N), l_u, mp_u))

l_0 = np.array([0.5])
mp_0 = np.array([0.4])
q1_0 = np.linspace(0, d, n)
q2_0 = np.linspace(np.pi, 0, n)
q3_0 = np.zeros(n)
q4_0 = np.zeros(n)
state_0 = np.vstack((q1_0, q2_0, q3_0, q4_0)).flatten()
u_0 = np.zeros(nu)
v0_i = np.concatenate((state_0, u_0))
v0 = np.concatenate((np.tile(v0_i, N), l_0, mp_0))

l_scaler = np.ones(1)
mp_scaler = np.ones(1)
state_scaler = np.ones(4 * n)
u_scaler = np.ones(nu) * 1e-1
x_scaler_i = np.concatenate((state_scaler, u_scaler))
x_scaler = np.concatenate((np.tile(x_scaler_i, N), l_scaler, mp_scaler))

o_scaler = 1e-2

jaxprob = JaxProblem(x0=v0, jax_obj=jax_obj, jax_con=jax_con, cl=cl, cu=cu, 
                     xl=xl, xu=xu, o_scaler=o_scaler, c_scaler=c_scaler, x_scaler=x_scaler)
optimizer = SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-8}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()
ans = optimizer.results['x'] / x_scaler


l = ans[-2]
mp = ans[-1]

print('l: ', l)
print('mp: ', mp)

for i in range(N):
    v_i = ans[i * ni:(i + 1) * ni]
    x_i = v_i[:4 * n].reshape((4, n))
    u_cp_i = v_i[4 * n:]
    u_i = bspline_comp(bspline_mtx, u_cp_i)

    position_i = x_i[0, :].flatten()
    velocity_i = x_i[2, :].flatten()
    angle_i = x_i[1, :].flatten()

    t = np.linspace(0, n*dt, n)
    t_ucp = np.linspace(0, n*dt, nu)
    plt.plot(t, angle_i, label='angle '+str(i))
    plt.plot(t, u_i, label='control '+str(i))
    plt.scatter(t_ucp, u_cp_i, label='control points '+str(i))
    plt.plot(t, position_i, label='position '+str(i))
    plt.plot(t, velocity_i, label='velocity '+str(i))
    plt.legend()

plt.show()





fig, axs = plt.subplots(1, N, figsize=(4 * N, 3))
axs = axs.flatten()

cart_width = 0.4
cart_height = 0.2
t = np.linspace(0, n * dt, n)

trajectory_data = []


for i in range(N):
    v_i = ans[i * ni:(i + 1) * ni]
    x_i = v_i[:4 * n].reshape((4, n))
    u_cp_i = v_i[4 * n:]
    u_i = bspline_comp(bspline_mtx, u_cp_i)

    position_i = x_i[0, :].flatten()
    velocity_i = x_i[2, :].flatten()
    angle_i = x_i[1, :].flatten()

    pole_x_i = position_i + l * np.sin(angle_i)
    pole_y_i = l * np.cos(angle_i)

    ax = axs[i]
    ax.set_aspect('equal')
    ax.grid(True)

    x_min, x_max = position_i.min() - l - 0.5, position_i.max() + l + 0.5
    y_min, y_max = -l - 0.2, l + 0.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'Trajectory {i + 1}')

    cart = plt.Rectangle(
        (position_i[0] - cart_width / 2, -cart_height / 2),
        cart_width,
        cart_height,
        facecolor='lightblue',
    )
    ax.add_patch(cart)

    pole_line, = ax.plot([], [], lw=5, color='red')
    pole_tip, = ax.plot([], [], 'o', color='green', markersize=12)

    trajectory_data.append((position_i, pole_x_i, pole_y_i, cart, pole_line, pole_tip))


x_min = min(position_i.min() - l - 0.5 for position_i, *_ in trajectory_data)
x_max = max(position_i.max() + l + 0.5 for position_i, *_ in trajectory_data)
y_min, y_max = -l - 0.2, l + 0.2

for ax in axs:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def animate(frame):
    animated_artists = []

    for position_i, pole_x_i, pole_y_i, cart, pole_line, pole_tip in trajectory_data:
        cart.set_xy((position_i[frame] - cart_width / 2, -cart_height / 2))
        x0, y0 = position_i[frame], 0
        pole_line.set_data([x0, pole_x_i[frame]], [y0, pole_y_i[frame]])
        pole_tip.set_data([pole_x_i[frame]], [pole_y_i[frame]])
        animated_artists.extend([cart, pole_line, pole_tip])

    return tuple(animated_artists)


ani = FuncAnimation(fig, animate, frames=len(t), blit=True, interval=30)

plt.show()