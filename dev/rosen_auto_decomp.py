from optiplex import DuPlex
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import IPOPT, PySLSQP
import numpy as np
import matplotlib.pyplot as plt
import pickle


objective = []
time = []

n = 200
N = 10

def make_sub_problem(subp, N, n):

        def sub_problem(x_init, y, mu, labels):

            recorder = csdl.Recorder(inline=True)
            recorder.start()

            x = csdl.Variable(value=np.zeros((n)))

            for j in range(n):
                xj = csdl.Variable(value=x_init[j])
                x = x.set(csdl.slice[j], xj)

                if labels[j]==subp:
                    xj.set_as_design_variable(scaler=1)


            coef = 100
            f = 0
            for i in csdl.frange(n - 1):
                f += coef * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2

            f.set_as_objective(scaler=1)
            recorder.stop()

            sim = csdl.experimental.JaxSimulator(recorder=recorder)
            prob = CSDLAlphaProblem(simulator=sim)
            # optimizer = SLSQP(prob, solver_options={'maxiter': 500, 'ftol': 1E-6}, turn_off_outputs=True)
            optimizer = PySLSQP(prob, solver_options={'maxiter': 500, 'acc': 1E-10}, turn_off_outputs=True)
            results = optimizer.solve()
            optimizer.print_results()

            objective.append(results['objective'])
            time.append(results['total_time'] + (time[-1] if len(time)>0 else 0))

            # new_x = []
            # k = 0
            # for j in range(n):
            #     if labels[j]==subp:
            #         print('Updating x_init at index:', j, 'with value:', results['x'][k])
            #         new_x.append(results['x'][k])
            #         k += 1
            #     else:
            #         new_x.append(x_init[j])

            k = 0
            for j in range(n):
                if labels[j]==subp:
                    # print('Updating x_init at index:', j, 'with value:', results['x'][k])
                    x_init[j] = results['x'][k]
                    k += 1

            return x_init
        
        # subP_functions.append(sub_problem)
        return sub_problem


subP_functions = []
for i in range(N):
    subPfunc = make_sub_problem(i, N, n)
    subP_functions.append(subPfunc)




def hessian(x):

    n = len(x)
    H = np.zeros((n, n))

    for i in range(n - 1):
        # Diagonal element H[i, i]
        H[i, i] += 1200 * x[i]**2 - 400 * x[i+1] + 2
        # Coupling with next variable
        H[i, i+1] += -400 * x[i]
        H[i+1, i] += -400 * x[i]
        # Contribution to next diagonal
        H[i + 1, i + 1] += 200

    return H

        

# size = int(n / N)
# v_init = []
# for i in range(N): v_init.append(np.zeros(size))
v_init = [0] * n


opt = DuPlex(blocks=subP_functions,
             hessian=hessian,
             x_init=v_init)

opt.solve(max_iter=300,
          tol=1e-5)

print('Solution: ', opt.x_init)
print("Success:", opt.success)
print('Iterations: ', opt.num_iter)
print('Time (s): ', opt.time)
print('Optimization time (s): ', time[-1])




with open('decomp_rosenbrock_n200_N2.pkl', 'wb') as f:
    data = {'objective': objective, 'time': time}
    pickle.dump(data, f)