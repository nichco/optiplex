import numpy as np
import modopt as mo

n = 100 # dimension

def jax_obj(v):

    f = 0
    for i in range(n - 1):
        f += 100 * (v[i + 1] - v[i]**2)**2 + (1 - v[i])**2

    return f
    
guess = np.array([-1.2, 1] * (n // 2))
x0 = guess
# x0 = np.zeros((n,))
jaxprob = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)

optimizer = mo.SLSQP(jaxprob, solver_options={'maxiter': 1000, 'ftol': 1e-7}, turn_off_outputs=True)
optimizer.solve()
optimizer.print_results()
ans = optimizer.results['x']