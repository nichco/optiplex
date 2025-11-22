import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, PySLSQP, SteepestDescent
import pandas as pd
import matplotlib.pyplot as plt

n = 200

# guess = np.array([-1.2, 1] * (n // 2))
# print(guess.shape)
# exit()

recorder = csdl.Recorder(inline=True)
recorder.start()

x = csdl.Variable(value=np.zeros((n)))
# x = csdl.Variable(value=guess)
x.set_as_design_variable(scaler=1)

f = 0
for i in csdl.frange(n - 1):
    f += 100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2

f.set_as_objective(scaler=1)
recorder.stop()


sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1e-6}, turn_off_outputs=True)
optimizer = PySLSQP(prob, solver_options={'maxiter': 6000, 'acc': 1e-10}, turn_off_outputs=True)
# optimizer = IPOPT(prob, solver_options={'max_iter': 10000, 'tol': 1e-10}, turn_off_outputs=True)
# optimizer = SteepestDescent(prob, maxiter=1000, opt_tol=1e-3, turn_off_outputs=True)
results = optimizer.solve()
optimizer.print_results()


# # parse the .out file
# filename = 'slsqp_summary.out'
# # Read the header line separately
# with open(filename, 'r') as file: headers = file.readline().strip().split()
# mf_df = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=headers)


# major = mf_df['MAJOR'].to_numpy()
# opt = mf_df['OPT']
# feas = mf_df['FEAS']
# obj = mf_df['OBJFUN'].to_numpy()


# plt.semilogy(major, obj, color='tab:blue')
# plt.show()