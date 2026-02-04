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
           x_init=v_init)


c = constraint(v_init)

opt.solve(max_outer_iter=100,
          max_inner_iter=10,
          ATOL_out=1e-5, 
          RTOL_out=1e-5,
          ATOL_in=1e-2, 
          RTOL_in=1e-2,
          ATOL_feas=1e-5,
          rho=1.1,
          mu=1.0,
          )



print('Time (s): ', opt.time)