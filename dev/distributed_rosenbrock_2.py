from optiplex import Plex
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import IPOPT, PySLSQP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

objective = []
time = []

n = 200
N = 2

def make_sub_problem(subp, N, n):

    def sub_problem(x_init, y, mu):

        recorder = csdl.Recorder(inline=True)
        recorder.start()

        x = csdl.Variable(value=np.zeros((n)))

        for j in range(N):
            start = j * n // N
            stop = (j + 1) * n // N
            xj = csdl.Variable(value=x_init[j])
            x = x.set(csdl.slice[start:stop], xj)

            if j==subp:
                xj.set_as_design_variable(scaler=1)

        coef = 200
        boef = 10
        f = 0
        for i in csdl.frange(n - 1):
            f += coef * (x[i + 1] - x[i]**2)**2 + (boef - x[i])**2

        f.set_as_objective(scaler=1)
        recorder.stop()

        sim = csdl.experimental.JaxSimulator(recorder=recorder)
        prob = CSDLAlphaProblem(simulator=sim)
        # optimizer = SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1E-6}, turn_off_outputs=True)
        optimizer = PySLSQP(prob, solver_options={'maxiter': 300, 'acc': 1E-10}, turn_off_outputs=True)
        results = optimizer.solve()
        optimizer.print_results()

        objective.append(results['objective'])
        time.append(results['total_time'] + (time[-1] if len(time)>0 else 0))

        x_init[subp] = results['x']
        # print(x_init)

        return x_init
    
    # subP_functions.append(sub_problem)
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



opt = Plex(blocks=subP_functions,
           x_init=v_init)

opt.solve(max_iter=300,
          tol=1e-9)

print('Solution: ', opt.x_init)
print("Success:", opt.success)
print('Iterations: ', opt.num_iter)
print('Time (s): ', opt.time)
print('Optimization time (s): ', time[-1])







plt.figure(figsize=(5,4))


# plt.semilogy(time, objective, color='tab:orange', linewidth=2, label='Distributed')
plt.plot(time, objective, color='tab:orange', linewidth=2, label='Distributed')

# plt.ylim(1e-5, 1e5)
plt.ylabel('Objective function value')
plt.xlabel('Wall time (s)')
plt.legend()
plt.show()