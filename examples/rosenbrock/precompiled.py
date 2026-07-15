from optiplex import BCD
import jax
import jax.numpy as jnp
import modopt as mo
import numpy as np
import time
import tracemalloc
import gc
import warnings
warnings.filterwarnings("ignore")

n = 1000  # dimension
N = 10    # number of subproblems

solution = np.ones(n)  # known solution for convergence criterion

if n % N:
    raise ValueError("n must be divisible by N")

block = n // N  # size of each subproblem's variable block

objective, times = [], []


def rosenbrock(x):
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def make_sub_problem(subp):

    start = subp * block  # static python int -> baked into the trace, not a jax value

    # --- pure function of (v, x_fixed) -- no closure over mutable x_init ---
    def sub_obj(v, x_fixed):
        x = jax.lax.dynamic_update_slice(x_fixed, v, (start,))
        return rosenbrock(x)

    # Compiled exactly ONCE per subproblem (first call triggers tracing).
    # Every later call with new x_fixed / v values of the SAME shape/dtype
    # reuses this compiled executable -- no retracing.
    val_and_grad = jax.jit(jax.value_and_grad(sub_obj, argnums=0))

    def sub_problem(x_init):

        x_fixed = jnp.concatenate(x_init)

        # Cache so that obj(v) and grad(v), which modopt/SLSQP call
        # separately, don't trigger two redundant compiled evaluations
        # at the same point.
        cache = {}

        def _evaluate(v):
            v = np.asarray(v)
            key = v.tobytes()
            if key not in cache:
                f, g = val_and_grad(jnp.asarray(v), x_fixed)
                cache.clear()  # keep only the most recent evaluation point
                cache[key] = (float(f), np.asarray(g))
            return cache[key]

        def obj(v):
            f, _ = _evaluate(v)
            return f

        def grad(v):
            _, g = _evaluate(v)
            return g

        v0 = np.asarray(x_init[subp])

        # Use modopt's ProblemLite -- a concrete container class that takes
        # plain obj/grad callables directly, instead of JaxProblem (which
        # would re-jit a fresh jax_obj internally on every call) or Problem
        # (which is abstract and requires subclassing).
        problem = mo.ProblemLite(
            x0=v0,
            obj=obj,
            grad=grad,
            xl=-np.inf,
            xu=np.inf,
        )

        optimizer = mo.SLSQP(problem, solver_options={'maxiter': 6000, 'ftol': 1e-7}, turn_off_outputs=True)
        # optimizer = mo.IPOPT(problem, solver_options={'max_iter': 6000, 'tol': 1e-7}, turn_off_outputs=True)

        t1 = time.perf_counter()
        optimizer.solve()
        t2 = time.perf_counter()
        opt_time = t2 - t1
        times.append(opt_time + (times[-1] if len(times) > 0 else 0))
        # optimizer.print_results()

        objective.append(optimizer.results['fun'])

        x_init[subp] = np.asarray(optimizer.results['x'])

        gc.collect()

        return x_init

    return sub_problem


# run the function generator to generate subproblems
subP_functions = []
for i in range(N):
    subPfunc = make_sub_problem(i)
    subP_functions.append(subPfunc)







size = int(n / N)
guess = np.array([-1.2, 1] * (n // 2))
v_init = []
# for i in range(N): v_init.append(np.zeros(size))
# for i in range(N): v_init.append(np.ones(size) * 3)
for i in range(N): v_init.append(guess[i*size:(i+1)*size])


opt = BCD(subproblems=subP_functions,
          x_init=v_init,
          solution=solution,
          eps=0.01, # percent solution tolerance
          )

tracemalloc.start()

opt.solve(max_iter=100)

current, peak = tracemalloc.get_traced_memory()
# print(f"Current: {current / 10**6:.2f} MB")
print(f"Peak: {peak / 10**6:.2f} MB")

tracemalloc.stop()

print('Solution: ', opt.x)
# print('Total time (s): ', opt.time)
print('Optimization time (s): ', times[-1])