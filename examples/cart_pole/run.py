from optiplex import Plex
import numpy as np
from examples.cart_pole.sub_problem import functions
from examples.cart_pole.constraint import constraint
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


opt = Plex(blocks=functions,
           constraint=constraint,
           x_init=v_init)

opt.solve(max_iter=100, 
          rho=1.2,
          tol=1e-7,
          itol=100,
          ctol=1e-4)



print('Success: ', opt.success)
print('Iterations: ', opt.k)
print('Time (s): ', opt.time)