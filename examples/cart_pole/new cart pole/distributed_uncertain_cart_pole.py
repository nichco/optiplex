import numpy as np
from optiplex import PlexC, Plex2, combo
from scipy.stats.qmc import LatinHypercube, scale
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from subproblem_generator import make_subproblem
import matplotlib.pyplot as plt

# sample the space of uncertain parameters using Latin hypercube sampling
# uncertain parameters: g, mu_cart, mu_pole
g_u = 9.81 + 0.1
g_l = 9.81 - 0.1
mu_cart_u = 0.03 + 0.01
mu_cart_l = 0.03 - 0.01
mu_pole_u = 0.03 + 0.01
mu_pole_l = 0.03 - 0.01
N = 2
sampler = LatinHypercube(d=3, seed=0)
samples = scale(sampler.random(N), 
                l_bounds=[g_l, mu_cart_l, mu_pole_l], 
                u_bounds=[g_u, mu_cart_u, mu_pole_u])
# samples = np.array([[9.81, 0.03, 0.03], [9.81, 0.03, 0.03]])
print(samples)

# parameters
n = 30
d = 0.8

subPfuns, opt_time = [], []
for i in range(N):
    subPfuns.append(make_subproblem(i, N, opt_time, samples))


def constraint(v):

    l_list = [x_init_i[0] for x_init_i in v]
    mp_list = [x_init_i[1] for x_init_i in v]

    l_constraint = combo(l_list)
    mp_constraint = combo(mp_list)

    return jnp.concatenate((l_constraint, mp_constraint))


l_0 = np.array([0.5])
mp_0 = np.array([0.4])
q1_0 = np.linspace(0, d, n)
q2_0 = np.linspace(np.pi, 0, n)
q3_0 = np.zeros(n)
q4_0 = np.zeros(n)
state_0 = np.vstack((q1_0, q2_0, q3_0, q4_0)).flatten()
u_0 = np.zeros(n)
v_i_0 = np.concatenate((l_0, mp_0, state_0, u_0))

v_init = [v_i_0 for _ in range(N)]


opt = Plex2(subproblems=subPfuns,
           con=constraint,
           tol=1e-5, # feasibility tolerance
           rho=1.3, # penalty increase factor
           mu=10.0, # initial penalty parameter
           max_mu=1e6, # maximum penalty parameter
           x_init=v_init,
           eps=1e-3, # initial inner loop tolerance
           eta=1e-5, # final inner loop tolerance
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=20,
          )

np.savez('uncertain_cart_pole_distributed_history_N2_V2.npz', history=opt.history, x_time=opt.x_time, feasibility=opt.feasibility, mu_history=opt.mu_history, multipliers=opt.y_history)

ans = opt.x
l_list = [a[0] for a in ans]
mp_list = [a[1] for a in ans]
print('l: ', l_list)
print('mp: ', mp_list)

solution = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_solution_N2.npz')
# solution = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_solution_N3.npz')
# solution = np.load('examples/cart_pole/new cart pole/uncertain_cart_pole_solution_N4.npz')
l_star = solution['l']
mp_star = solution['mp']
x_star = np.array(solution['x_list'])
u_star = np.array(solution['u_list'])

params = np.broadcast_to([l_star, mp_star], (N, 2))
solution = np.concatenate((params, x_star, u_star), axis=1).ravel()

h = np.array(opt.history)

error = np.linalg.norm(h.reshape(len(h), -1) - solution, axis=1)

plt.semilogy(error, linewidth=2)
plt.grid()
plt.show()