import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from subproblems_with_bsplines_and_stress import make_subproblem
from optiplex import PlexC, Plex2, combo
import tracemalloc

num = 2 # number of operating conditions
sampler = LatinHypercube(d=2, seed=42)
samples = scale(sampler.random(num), l_bounds=[0.4, 180], u_bounds=[0.6, 220])
# samples = [(0.4135, 210), (0.4135, 207)] # test solution a
# samples = [(0.4135, 210), (0.4135, 210)] # test solution b
# samples = [(0.4135, 210), (0.5, 191)] # test solution c
# samples = [(0.4226044, 191.2224312), (0.51414021, 206.05263942)] # test solution d
print(samples)

ns = 45 # number of spanwise panels

# generate subproblem functions
subPfuns, opt_time = [], []
for i, (rho_atm, v_inf) in enumerate(samples): 
    subPfuns.append(make_subproblem(i, num, opt_time, samples))



def spy():


    return




n_twist_cp = 17
n_thickness_cp = 10

alpha_i_init = np.array([0])
twist_cp_i_init = np.ones(n_twist_cp) * np.deg2rad(5)
thickness_cp_i_init = np.ones(n_thickness_cp) * 0.002

x_i_init = np.concatenate([alpha_i_init, twist_cp_i_init, thickness_cp_i_init])

x_init = [x_i_init for _ in range(num)]




def con(v_init):

    # alphas = [x_init_i[0] for x_init_i in v_init]
    twist_cps = [x_init_i[1:1 + n_twist_cp] for x_init_i in v_init]
    thickness_cps = [x_init_i[1 + n_twist_cp:] for x_init_i in v_init]

    twist_cp_constraint = combo(twist_cps)
    # print('twist_cp_constraint shape: ', twist_cp_constraint.shape)
    thickness_cp_constraint = combo(thickness_cps)
    # print('thickness_cp_constraint shape: ', thickness_cp_constraint.shape)

    # return jnp.concatenate((0.125*twist_cp_constraint, 2*thickness_cp_constraint))
    return jnp.concatenate((0.125*twist_cp_constraint, 1*thickness_cp_constraint))



tracemalloc.start()

# opt = PlexC(subproblems=subPfuns,
#             x_init=x_init,
#             con=con,
#             mu=10,
#             max_mu=1e5,
#             rho=1.5,
#             tau=0.5,
#             tol=1e-4, # outer loop feasibility
#             eps=1e-4, # inner loop convergence
#             )

opt = Plex2(subproblems=subPfuns,
            x_init=x_init,
            con=con,
            mu=np.ones(n_twist_cp + n_thickness_cp) * 1, # N=2
            # mu=np.ones((num - 1) * (n_twist_cp + n_thickness_cp)) * 1, # N=3
            # mu=np.ones((135)) * 1, # N=4
            # mu=np.ones((378)) * 1, # N=6
            # mu=np.ones((729)) * 1, # N=8
            # mu=np.ones((1188)) * 1, # N=10
            max_mu=1e6,
            rho=1.5,
            tau=0.5,
            tol=1e-4, # outer loop feasibility
            eps=1e-2, # initial inner loop convergence
            eta=1e-3,#1e-4, # final inner loop convergence
            )

opt.solve(max_outer_iter=100, 
          max_inner_iter=100,
          )

print('Total time (s): ', opt.tf)
print('Optimization time (s): ', opt_time[-1])

# print peak memory usage
_, peak = tracemalloc.get_traced_memory()
print(f"Peak: {peak / 10**6}MB")
tracemalloc.stop()


x = opt.x
alphas = []
twist_cps = []
thickness_cps = []
for i in range(num):
    x_i = x[i]
    alphas.append(x_i[0])
    twist_cps.append(x_i[1:1 + n_twist_cp])
    thickness_cps.append(x_i[1 + n_twist_cp:])


print('alphas (deg): ', np.rad2deg(alphas))

# print the total optimization time
print('Total optimization time (s): ', opt_time[-1])



# save history to an npz file
# np.savez('examples/crm_aero_struct/distributed_solution_N3_V2.npz', history=opt.history, time=opt.x_time, feasibility=opt.feasibility, mu_history=opt.mu_history, multipliers=opt.y_history)


# solution = np.load('examples/crm_aero_struct/test_solution_a.npz')
# solution = np.load('examples/crm_aero_struct/test_solution_b.npz')
# solution = np.load('examples/crm_aero_struct/test_solution_c.npz')
# solution = np.load('examples/crm_aero_struct/solution_num_2_bsplines_and_stress.npz')
solution = np.load('examples/crm_aero_struct/test_solution_d.npz')
# solution = np.load('examples/crm_aero_struct/test_solution_N6.npz')
alphas_star = solution['alphas']
print('alphas_star (deg): ', np.rad2deg(alphas_star))
twist_cp_star = solution['twist_cp']
thickness_cp_star = solution['thickness_cp']

from jax_b_splines import get_bspline_mtx, bspline_comp
n_twist_cp = 17
twist_bspline_mtx = get_bspline_mtx(n_twist_cp, ns)

n_thickness_cp = 10
thickness_bspline_mtx = get_bspline_mtx(n_thickness_cp, ns - 1)

twist_star = bspline_comp(twist_bspline_mtx, twist_cp_star)
thickness_star = bspline_comp(thickness_bspline_mtx, thickness_cp_star)


for i in range(num):
    twist_i = bspline_comp(twist_bspline_mtx, twist_cps[i])
    plt.plot(twist_i, label=f'twist {i}')
    plt.scatter(np.linspace(0, ns - 1, n_twist_cp), twist_cps[i], marker='o', label=f'twist cp {i}')

plt.plot(twist_star, label='twist solution', linestyle='--', linewidth=2)
plt.scatter(np.linspace(0, ns - 1, n_twist_cp), twist_cp_star, marker='o', label='twist cp solution', color='tab:green')
plt.legend()
plt.title('Twist')
plt.show()

for i in range(num):
    thickness_i = bspline_comp(thickness_bspline_mtx, thickness_cps[i])
    plt.plot(thickness_i, label=f'thickness {i}')
    plt.scatter(np.linspace(0, ns - 2, n_thickness_cp), thickness_cps[i], marker='o', label=f'thickness cp {i}')

plt.plot(thickness_star, label='thickness solution', linestyle='--', linewidth=2)
plt.scatter(np.linspace(0, ns - 2, n_thickness_cp), thickness_cp_star, marker='o', label='thickness cp solution', color='tab:green')
plt.legend()
plt.title('Thickness')
plt.show()


vars = np.array(opt.history)
print('vars shape: ', vars.shape) # should be (n, num, len(x_i))
vars = vars.reshape(vars.shape[0], -1) # reshape to (n, num * len(x_i)) for easier plotting

# plt.plot(vars)
# plt.show()

plt.semilogy(np.abs(vars))
plt.show()

# normalize vars for better visualization
vars_norm = vars / np.max(vars, axis=0)
plt.plot(vars_norm)
plt.show()




alphas_star = np.array(solution['alphas']).reshape(-1, 1)
twist_cp_star = solution['twist_cp']
thickness_cp_star = solution['thickness_cp']

twist_params = np.broadcast_to(twist_cp_star, (num, n_twist_cp))
thickness_params = np.broadcast_to(thickness_cp_star, (num, n_thickness_cp))
solution = np.concatenate((alphas_star, twist_params, thickness_params), axis=1).ravel()

h = np.array(opt.history)

error = np.zeros(len(h))
for i in range(len(h)):
    h_i = h[i].flatten()
    error_i = np.linalg.norm((h_i - solution))
    error[i] = error_i

plt.semilogy(error, linewidth=2)

plt.ylabel('Error')
plt.xlabel('Iteration')
plt.show()


# N = 2
# time = 15.92

# N = 6
# time = 52.76

# N = 10
# time = 107.42