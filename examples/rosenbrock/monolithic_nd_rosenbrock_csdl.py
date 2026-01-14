import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP, IPOPT, PySLSQP, SteepestDescent

n = 200

# guess = np.array([-1.2, 1] * (n // 2))

recorder = csdl.Recorder(inline=True)
recorder.start()

x = csdl.Variable(value=np.zeros((n)))
# x = csdl.Variable(value=guess)
x.set_as_design_variable(scaler=1)

f = csdl.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

f.set_as_objective(scaler=1)
recorder.stop()


sim = csdl.experimental.JaxSimulator(recorder=recorder)
prob = CSDLAlphaProblem(simulator=sim)
optimizer = SLSQP(prob, solver_options={'maxiter': 1000, 'ftol': 1e-6}, turn_off_outputs=True)
# optimizer = PySLSQP(prob, solver_options={'maxiter': 6000, 'acc': 1e-10}, turn_off_outputs=True)
# optimizer = IPOPT(prob, solver_options={'max_iter': 10000, 'tol': 1e-10}, turn_off_outputs=True)
# optimizer = SteepestDescent(prob, maxiter=1000, opt_tol=1e-3, turn_off_outputs=True)
results = optimizer.solve()
optimizer.print_results()