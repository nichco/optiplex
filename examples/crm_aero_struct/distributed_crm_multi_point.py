import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube, scale
from subproblem_functions import make_subproblem
from optiplex import PlexC, Plex2, combo






num = 2 # number of operating conditions
# sampler = LatinHypercube(d=2, seed=42)
# samples = scale(sampler.random(num), l_bounds=[0.4, 180], u_bounds=[0.6, 220])
samples = [(0.4135, 210), (0.4135, 190)]
print(samples)


ns = 33 # number of spanwise panels
si = np.concatenate(([10], np.ones(ns), np.ones(ns - 1) * 100))

cs = np.concatenate((np.ones(ns) * 10, np.ones(ns - 1) * 20)) * 1e-2 # 2e-1 # constraint scaling for better conditioning of the dual updates

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

    c = jnp.concatenate((twist_constraint, thickness_constraint)) * cs

    return c



# opt = PlexC(subproblems=subPfuns,
#             x_init=x_init,
#             con=con,
#             mu=0.1,#1,#10,
#             max_mu=1e5,
#             rho=1.5,
#             tau=0.5,
#             tol=1e-3, # outer loop feasibility
#             eps=1e-4, # inner loop convergence
#             )

opt = Plex2(subproblems=subPfuns,
            x_init=x_init,
            con=con,
            mu=10,
            max_mu=1e5,
            rho=1.1,
            tau=0.5,
            tol=1e-3, # outer loop feasibility
            eps=1e-2, # inner loop convergence
            eta=1e-4, # final inner loop convergence
            )

opt.solve(max_outer_iter=300, 
          max_inner_iter=100,
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
