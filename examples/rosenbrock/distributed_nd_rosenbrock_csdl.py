from optiplex import Plex
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import IPOPT, SLSQP
import numpy as np
import matplotlib.pyplot as plt
import time

objective = []
times = []

n = 200
N = 2

def make_sub_problem(subp, N, n):

    def sub_problem(x_init, y, mu):

        recorder = csdl.Recorder(inline=True)
        recorder.start()

        x = csdl.Variable(value=np.zeros((n)))

        for j in range(N):
            xj = csdl.Variable(value=x_init[j])
            x = x.set(csdl.slice[j * n // N : (j + 1) * n // N], xj)

            if j==subp:
                xj.set_as_design_variable(scaler=1)

        f = csdl.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

        f.set_as_objective(scaler=1)
        recorder.stop()

        sim = csdl.experimental.JaxSimulator(recorder=recorder)
        prob = CSDLAlphaProblem(simulator=sim)
        optimizer = SLSQP(prob, solver_options={'maxiter': 500, 'ftol': 1E-6}, turn_off_outputs=True)
        # optimizer = PySLSQP(prob, solver_options={'maxiter': 500, 'acc': 1E-10}, turn_off_outputs=True)
        t1 = time.perf_counter()
        results = optimizer.solve()
        t2 = time.perf_counter()
        opt_time = t2 - t1

        optimizer.print_results()

        # objective.append(results['objective'])
        objective.append(results['fun'])
        # time.append(results['total_time'] + (time[-1] if len(time)>0 else 0))
        times.append(opt_time + (times[-1] if len(times)>0 else 0))

        x_init[subp] = results['x']

        return x_init
    
    return sub_problem


subP_functions = []
for i in range(N):
    subPfunc = make_sub_problem(i, N, n)
    subP_functions.append(subPfunc)



# check to make sure n is divisible by N
if n % N != 0:
    raise ValueError("n must be divisible by N")

size = int(n / N)

v_init = []
for i in range(N): v_init.append(np.zeros(size))
# for i in range(N): v_init.append(np.ones(size) * -1)



opt = Plex(blocks=subP_functions,
           x_init=v_init)

opt.solve(max_iter=300,
          tol=1e-5)

print('Solution: ', opt.x_init)
print("Success:", opt.success)
print('Iterations: ', opt.num_iter)
print('Time (s): ', opt.time)
print('Optimization time (s): ', times[-1])



# objective.insert(0, 199.0)
# time.insert(0, 0.0)