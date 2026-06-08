import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from subproblem_functions import make_subproblem
from optiplex import PlexC, Plex2, combo






num = 2 # number of operating conditions
sampler = LatinHypercube(d=2, seed=42)
samples = scale(sampler.random(num), l_bounds=[0.4, 190], u_bounds=[0.6, 220])
# samples = [(0.4135, 210), (0.4135, 190)]
print(samples)


ns = 33 # number of spanwise panels
si = np.concatenate(([10], np.ones(ns), np.ones(ns - 1) * 100))

cs = 6e-2 # constraint scaling for better conditioning of the dual updates

# generate subproblem functions
subPfuns, opt_time = [], []
for i, (rho_atm, v_inf) in enumerate(samples): 
    subPfuns.append(make_subproblem(i, rho_atm, v_inf, num, si, cs, opt_time))



alpha_i_init = np.array([0])
twist_i_init = np.ones(ns) * np.deg2rad(5)
thickness_i_init = np.ones(ns - 1) * 0.002

x_i_init = np.concatenate([alpha_i_init, twist_i_init, thickness_i_init]) * si

x_init = [x_i_init for _ in range(num)]




def con(v_init):
    alphas = []
    twists = []
    thicknesses = []
    for i in range(num):
        # x_init_i = v_init[i]
        x_init_i = v_init[i] / si # unscale the decision variables
        alphas.append(x_init_i[0])
        twists.append(x_init_i[1:1 + ns])
        thicknesses.append(x_init_i[1 + ns:])

    twist_constraint = combo(twists) # modified combo to remove one pair
    thickness_constraint = combo(thicknesses) # modified combo to remove one pair

    c = jnp.concatenate((twist_constraint, 10 * thickness_constraint)) * cs

    return c



opt = PlexC(subproblems=subPfuns,
            x_init=x_init,
            con=con,
            mu=1,#10,
            max_mu=1e5,
            rho=1.5,
            tau=0.5,
            tol=1e-4, # outer loop feasibility
            eps=1e-2, # inner loop convergence
            )

# opt = Plex2(subproblems=subPfuns,
#             x_init=x_init,
#             con=con,
#             mu=10,
#             max_mu=1e5,
#             rho=1.5,
#             tau=0.5,
#             tol=1e-3, # outer loop feasibility
#             eps=1e-2, # inner loop convergence
#             eta=1e-4, # final inner loop convergence
#             )

opt.solve(max_outer_iter=4, 
          max_inner_iter=3,
          )

x = opt.x
alphas = []
twists = []
thicknesses = []
for i in range(num):
    x_i = x[i] / si # unscale the variables
    alphas.append(x_i[0])
    twists.append(x_i[1:1 + ns])
    thicknesses.append(x_i[1 + ns:])


print('alphas: ', alphas)
print('twists: ', twists)
print('thicknesses: ', thicknesses)

# print the total optimization time
print('Total optimization time (s): ', opt_time[-1])


for i in range(num):
    plt.plot(twists[i], label=f'twist {i}')
plt.legend()
plt.title('Twist')
plt.show()

for i in range(num):
    plt.plot(thicknesses[i], label=f'thickness {i}')
plt.legend()
plt.title('Thickness')
plt.show()


vars = np.array(opt.history)
print('vars shape: ', vars.shape) # should be (n, num, len(x_i))
vars = vars.reshape(vars.shape[0], -1) # reshape to (n, num * len(x_i)) for easier plotting

# plt.plot(vars)
# plt.show()

# vars = np.abs(vars)
# plt.semilogy(vars)
# plt.show()

# normalize vars for better visualization
vars_norm = vars / np.max(vars, axis=0)
plt.plot(vars_norm)
plt.show()





solution = np.load('examples/crm_aero_struct/solution.npz')
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