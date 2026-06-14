import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from subproblem_functions import make_subproblem
from optiplex import Plex, PlexC, Plex2, combo






num = 2 # number of operating conditions
sampler = LatinHypercube(d=2, seed=42)
samples = scale(sampler.random(num), l_bounds=[0.4, 190], u_bounds=[0.6, 220])
# samples = [(0.4135, 210), (0.4135, 207)] # test solution a
# samples = [(0.4135, 210), (0.4135, 210)] # test solution b
print(samples)


ns = 33 # number of spanwise panels

cs = 1e-1 # constraint scaling for better conditioning of the dual updates
# cs = 2e-1 # constraint scaling for better conditioning of the dual updates

# generate subproblem functions
subPfuns, opt_time = [], []
for i, (rho_atm, v_inf) in enumerate(samples): 
    # subPfuns.append(make_subproblem(i, rho_atm, v_inf, num, si, cs, opt_time, samples))
    subPfuns.append(make_subproblem(i, num, cs, opt_time, samples))



alpha_i_init = np.array([0])
twist_i_init = np.ones(ns) * np.deg2rad(5)
thickness_i_init = np.ones(ns - 1) * 0.002

x_i_init = np.concatenate([alpha_i_init, twist_i_init, thickness_i_init])

x_init = [x_i_init for _ in range(num)]




def con(v_init):
    alphas = []
    twists = []
    thicknesses = []
    for i in range(num):
        x_init_i = v_init[i]
        alphas.append(x_init_i[0])
        twists.append(x_init_i[1:1 + ns])
        thicknesses.append(x_init_i[1 + ns:])

    twist_constraint = combo(twists) # modified combo to remove one pair

    # offset all thicknesses
    thicknesses = [thickness + 0.1 for thickness in thicknesses]

    thickness_constraint = combo(thicknesses) # modified combo to remove one pair

    c = jnp.concatenate((twist_constraint, thickness_constraint)) * cs

    return c



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
            mu=1,
            # mu=np.ones(ns + ns - 1) * 1,
            max_mu=1e7,
            rho=1.2,
            tau=0.5,
            tol=1e-3, # outer loop feasibility
            eps=1e-2, # initial inner loop convergence
            eta=0.5e-4, # final inner loop convergence
            )

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
twists = []
thicknesses = []
for i in range(num):
    x_i = x[i]
    alphas.append(x_i[0])
    twists.append(x_i[1:1 + ns])
    thicknesses.append(x_i[1 + ns:])


print('alphas (deg): ', np.rad2deg(alphas))
# print('twists: ', twists)
# print('thicknesses: ', thicknesses)

# print the total optimization time
print('Total optimization time (s): ', opt_time[-1])



# solution = np.load('examples/crm_aero_struct/test_solution_a.npz')
solution = np.load('examples/crm_aero_struct/test_solution_lhs_num_2.npz')
alphas_star = solution['alphas']
twist_star = solution['twist']
thickness_star = solution['thickness']
samples_star = solution['samples']




for i in range(num):
    plt.plot(twists[i], label=f'twist {i}')

plt.plot(twist_star, label='twist solution', linestyle='--', linewidth=2)
plt.legend()
plt.title('Twist')
plt.show()

for i in range(num):
    plt.plot(thicknesses[i], label=f'thickness {i}')

plt.plot(thickness_star, label='thickness solution', linestyle='--', linewidth=2)
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
        x_j_star = np.concatenate(([alphas_star[j]], twist_star, thickness_star))
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