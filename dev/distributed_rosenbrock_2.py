from optiplex import Plex, WarmPlex
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import IPOPT, PySLSQP
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle



# # compute obj init
# x = np.zeros((200))
# coef = 100
# f = 0
# for i in range(200 - 1):
#     f += coef * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2

# print('Initial objective: ', f)
# exit()

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

            coef = 100
            f = 0
            for i in csdl.frange(n - 1):
                f += coef * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2

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





def problem(x_init, y, mu):

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    x = csdl.Variable(value=np.zeros((n)))

    for j in range(N):
        start = j * n // N
        stop = (j + 1) * n // N
        xj = csdl.Variable(value=x_init[j])
        x = x.set(csdl.slice[start:stop], xj)

        xj.set_as_design_variable(scaler=1)

    coef = 100
    f = 0
    for i in csdl.frange(n - 1):
        f += coef * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2

    f.set_as_objective(scaler=1)
    recorder.stop()

    sim = csdl.experimental.JaxSimulator(recorder=recorder)
    prob = CSDLAlphaProblem(simulator=sim)
    # optimizer = SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1E-6}, turn_off_outputs=True)
    optimizer = PySLSQP(prob, solver_options={'maxiter': 600, 'acc': 1E-10}, turn_off_outputs=True)
    results = optimizer.solve()
    optimizer.print_results()

    objective.append(results['objective'])
    time.append(results['total_time'] + (time[-1] if len(time)>0 else 0))

    for i in range(N):
        x_init[i] = results['x'][i*(n//N):(i+1)*(n//N)]

    return x_init
        


# check to make sure n is divisible by N
if n % N != 0:
    raise ValueError("n must be divisible by N")

size = int(n / N)

v_init = []
for i in range(N): v_init.append(np.zeros(size))



# opt = Plex(blocks=subP_functions,
#            x_init=v_init)

opt = WarmPlex(blocks=subP_functions,
               problem=problem,
               x_init=v_init)

opt.solve(max_iter=4,
          tol=1e-9)

print('Solution: ', opt.x_init)
print("Success:", opt.success)
print('Iterations: ', opt.num_iter)
print('Time (s): ', opt.time)
print('Optimization time (s): ', time[-1])


# add initial objective of 199 to objective list
objective.insert(0, 199.0)
time.insert(0, 0.0)



# parse the .out file
filename = 'monolithic_200.out'
# Read the header line separately
with open(filename, 'r') as file: headers = file.readline().strip().split()
mf_df = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=headers)

monolithic_major = mf_df['MAJOR'].to_numpy()
monolithic_optimality = mf_df['OPT']
monolithic_feasibility = mf_df['FEAS']
monolithic_objective = mf_df['OBJFUN'].to_numpy()

# monolithic_time_200 = 6.602
monolithic_time_200 = 3.499
# monolithic_time_400 = 71.98
monolithic_time = np.linspace(0, monolithic_time_200, len(monolithic_major))



plt.figure(figsize=(5,4))

plt.semilogy(monolithic_time, monolithic_objective, color='tab:blue', linewidth=2, label='Monolithic')
# plt.plot(monolithic_time, monolithic_objective, color='tab:blue', linewidth=2, label='Monolithic')

plt.semilogy(time[:-2], objective[:-2], color='tab:orange', linewidth=2, label='Distributed', marker='s')
plt.semilogy(time[-2:], objective[-2:], color='tab:green', linewidth=2, label='Warm-Start Monolithic', marker='o', linestyle='--')

plt.ylabel('Objective Function Value')
plt.xlabel('Wall Time (s)')
plt.legend()
# plt.ylim(1e-12, 1e5)
plt.grid(color='lavender', alpha=0.5)

plt.savefig('rosenbrock_warmstart.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()