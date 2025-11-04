from optiplex import Plex
import numpy as np
from sub_problem import make_sub_problem
from constraint import constraint

N = 2
mission_ranges = np.linspace(1000, 3000, N)

subP_functions = []
for i in range(N):
    subPfunc = make_sub_problem(mission_ranges[i], i)
    subP_functions.append(subPfunc)


nu = 60
eta_scale = 1.0
theta_scale = 1e1
tf_scale = 1e-3
b_scale = 1e-1
eta0 = np.linspace(0.6, 0.2, nu) * eta_scale
theta0 = np.linspace(np.deg2rad(3), np.deg2rad(1), nu) * theta_scale
tf0 = np.array([7000.0]) * tf_scale
b0 = np.array([32.0]) * b_scale

v_init = []
# automate the construction of v_init for changing N
for i in range(N): 
    v_init.append(eta0)
    v_init.append(theta0)
    v_init.append(tf0)
    v_init.append(b0)



opt = Plex(blocks=subP_functions,
           constraint=constraint,
           x_init=v_init)

opt.mu = 1e-5

opt.solve(max_iter=2, 
          rho=1.2,
          tol=1e-7,
          ctol=1e-4)

print("Success:", opt.success)