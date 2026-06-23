import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from subproblems_with_bsplines_and_stress import make_subproblem
from optiplex import Plex, PlexC, Plex2, combo






num = 2 # number of operating conditions
# sampler = LatinHypercube(d=2, seed=42)
# samples = scale(sampler.random(num), l_bounds=[0.4, 180], u_bounds=[0.6, 220])
# samples = [(0.4135, 210), (0.4135, 207)] # test solution a
# samples = [(0.4135, 210), (0.4135, 210)] # test solution b
# samples = [(0.4135, 210), (0.5, 191)] # test solution c
samples = [(0.4226044, 191.2224312), (0.51414021, 206.05263942)] # test solution d
print(samples)
# exit()

ns = 45 # number of spanwise panels

# generate subproblem functions
subPfuns, opt_time = [], []
for i, (rho_atm, v_inf) in enumerate(samples): 
    # subPfuns.append(make_subproblem(i, rho_atm, v_inf, num, si, cs, opt_time, samples))
    subPfuns.append(make_subproblem(i, num, opt_time, samples))

n_twist_cp = 17
n_thickness_cp = 10

alpha_i_init = np.array([0])
twist_cp_i_init = np.ones(n_twist_cp) * np.deg2rad(5)
thickness_cp_i_init = np.ones(n_thickness_cp) * 0.002

x_i_init = np.concatenate([alpha_i_init, twist_cp_i_init, thickness_cp_i_init])

x_init = [x_i_init for _ in range(num)]




def con(v_init):

    alphas = [x_init_i[0] for x_init_i in v_init]
    twist_cps = [x_init_i[1:1 + n_twist_cp] for x_init_i in v_init]
    thickness_cps = [x_init_i[1 + n_twist_cp:] for x_init_i in v_init]

    twist_cp_constraint = combo(twist_cps)
    thickness_cp_constraint = combo(thickness_cps)

    return jnp.concatenate((0.125*twist_cp_constraint, 2*thickness_cp_constraint))



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
            # mu=1,
            mu=np.ones(n_twist_cp + n_thickness_cp) * 1,
            max_mu=1e6,
            rho=1.2,
            tau=0.5,
            tol=1e-4, # outer loop feasibility
            eps=1e-3, # initial inner loop convergence
            eta=1e-4, # final inner loop convergence
            )

# best error: 0.1226

opt.solve(max_outer_iter=100, 
          max_inner_iter=100,
          )

print('Total time (s): ', opt.tf)

# opt = Plex(subproblems=subPfuns,
#            x_init=x_init,
#            con=con,
#            tol=1e-5, # outer loop feasibility
#            mu=1,
#            max_mu=1e5,
#            rho=1.5,
#            )

# opt.solve(max_inner_iter=100,
#           max_outer_iter=100,
#           eps_inner=1e-3,
#           eps_outer=1e-5,
#           )

# print('Total time (s): ', opt.time)

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
# print('twists: ', twist_cps)
# print('thicknesses: ', thickness_cps)

# print the total optimization time
print('Total optimization time (s): ', opt_time[-1])



# solution = np.load('examples/crm_aero_struct/test_solution_a.npz')
# solution = np.load('examples/crm_aero_struct/test_solution_b.npz')
# solution = np.load('examples/crm_aero_struct/test_solution_c.npz')
# solution = np.load('examples/crm_aero_struct/solution_num_2_bsplines_and_stress.npz')
solution = np.load('examples/crm_aero_struct/test_solution_d.npz')
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







vars = np.array(opt.history)
n = vars.shape[0]

error = []
for i in range(n):
    sol = []
    x_i = []
    for j in range(num):
        x_j_star = np.concatenate(([alphas_star[j]], twist_cp_star, thickness_cp_star))
        sol.append(x_j_star)

        x_i_j = vars[i, j, :]
        x_i.append(x_i_j)

    sol = np.concatenate(sol)
    x_i = np.concatenate(x_i)

    error_i = np.linalg.norm((x_i - sol) / sol)
    error.append(error_i)

plt.semilogy(opt.x_time, error)
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.show()

print('Final error: ', error[-1])

    
# x_star = np.concatenate([solution['twist'], solution['thickness']])

# history_vecs = [np.concatenate(h[:2]) for h in opt.history]
# error = [np.linalg.norm((x - x_star) / x_star) for x in history_vecs]

# print('CD: ', cd_history[-1])

# # plt.semilogy(error)
# # plt.xlabel('Iteration')
# # plt.ylabel('Relative error')
# # plt.show()

# plt.semilogy(opt.x_time, error)
# plt.xlabel('Time (s)')
# plt.ylabel('Relative error')
# plt.show()