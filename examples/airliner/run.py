from optiplex import Plex
import numpy as np
from sub_problem import make_sub_problem
from constraint import constraint
import matplotlib.pyplot as plt

N = 2
# mission_ranges = np.linspace(1.5e6, 2e6, N)
mission_ranges = np.array([2e6, 2e6])

subP_functions = []
for i in range(N):
    subPfunc = make_sub_problem(mission_ranges[i], i)
    subP_functions.append(subPfunc)


nu = 60
eta0 = np.linspace(0.6, 0.2, nu)
theta0 = np.linspace(np.deg2rad(3), np.deg2rad(1), nu)
tf0 = np.array([7000.0])
b0 = np.array([32.0])

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

opt.mu = 1e-2

opt.solve(max_iter=30, 
          rho=1.5,
          tol=1e-2,
          ctol=1e-3)

print("Success:", opt.success)
print('Iterations: ', opt.num_iter)
print('Time (s): ', opt.time)


diffs = opt.diffs
constraint_violations = opt.constraint_violations


plt.semilogy(diffs, label='Max Relative Diff')
plt.semilogy(constraint_violations, label='Max Constraint Violation')
plt.xlabel('Iteration', fontsize=14)

plt.show()
