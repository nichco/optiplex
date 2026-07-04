from optiplex import Plex
import numpy as np
from examples.cart_pole.problem_definition import functions, constraint
import warnings
warnings.filterwarnings("ignore")

N = 2

q1_0 = np.linspace(0, 0.8, 30)
q2_0 = np.linspace(np.pi, 0, 30)
q3_0 = np.zeros(30)
q4_0 = np.zeros(30)
x0 = np.vstack((q1_0, q2_0, q3_0, q4_0))
u0 = np.zeros((30))
l0 = 0.5
mp0 = 0.4

v_init = []
# automate the construction of v_init for changing N
for i in range(N): v_init.append(l0)
for i in range(N): v_init.append(mp0)
for i in range(N): v_init.append(x0)
for i in range(N): v_init.append(u0)


opt = Plex(subproblems=functions,
           con=constraint,
           tol=1e-5, # feasibility tolerance
           rho=1.1, # penalty increase factor
           mu=1.0, # initial penalty parameter
           max_mu=1e6, # maximum penalty parameter
           x_init=v_init,
           )

opt.solve(max_outer_iter=100,
          max_inner_iter=10,
          eps_inner=1e-2, # inner loop tolerance
          eps_outer=1e-5, # outer loop tolerance
          )

print('Time (s): ', opt.time)