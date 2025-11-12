import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, PySLSQP
from mod_steepest_descent import SteepestDescent
from scipy.stats import qmc

n = 10

recorder = csdl.Recorder(inline=True)
recorder.start()

# x = csdl.Variable(value=np.zeros((n)))
x = csdl.Variable(value=np.ones((n)))
x.set_as_design_variable(scaler=1)

# coef = 100
acoef = csdl.Variable(value=100.0)
bcoef = csdl.Variable(value=1.0)
f = 0
for i in csdl.frange(n - 1):
    f += acoef * (x[i + 1] - x[i]**2)**2 + (bcoef - x[i])**2

f.set_as_objective(scaler=1)

recorder.stop()



# inputs = [x, acoef, bcoef]
# # outputs = [CL, CDi, Cp, L, Di]

# sim = csdl.experimental.JaxSimulator(
#     recorder=recorder,
#     additional_inputs=inputs,
#     # additional_outputs = outputs,
#     gpu=False
# )


# sim = csdl.experimental.JaxSimulator(recorder=recorder)

N = 10
sampler = qmc.LatinHypercube(d=n, rng=0)
sample = sampler.random(n=N)

l_bounds = [-10] * n
u_bounds = [10] * n
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
# print(sample_scaled)
# print(sample_scaled.shape)

# coef_values = np.linspace(10, 1000, N)

for i in range(N):

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    # x = csdl.Variable(value=np.zeros((n)))
    x = csdl.Variable(value=sample_scaled[i, :])
    x.set_as_design_variable(scaler=1)

    acoef = csdl.Variable(value=100)
    bcoef = csdl.Variable(value=1.0)
    f = 0
    for i in csdl.frange(n - 1):
        f += acoef * (x[i + 1] - x[i]**2)**2 + (bcoef - x[i])**2

    f.set_as_objective(scaler=1)

    recorder.stop()


    sim = csdl.experimental.JaxSimulator(recorder=recorder)
    prob = CSDLAlphaProblem(simulator=sim)
    # optimizer = SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1E-6}, turn_off_outputs=True)
    # optimizer = PySLSQP(prob, solver_options={'maxiter': 1000, 'acc': 1E-10}, turn_off_outputs=True)
    optimizer = SteepestDescent(prob, maxiter=1000, opt_tol=1e-3, turn_off_outputs=True)

    try:
        results = optimizer.solve()
        optimizer.print_results()
    except:
        print("Optimization failed for this case.")
        continue


    # print(x.value)