from optiplex import BCD
import jax
import jax.numpy as jnp
import modopt as mo
import numpy as np
import time
import tracemalloc
# import gc
import warnings
warnings.filterwarnings("ignore")

# https://www.sfu.ca/~ssurjano/dixonpr.html

n = 100  # dimension
N = 2    # number of subproblems

i = np.arange(1, n + 1)
solution = 2**(-(2**i - 2) / (2**i))

block = n // N  # size of each subproblem's variable block

objective, times = [], []

# weight vector for the Dixon-Price sum term (fixed for a given n, so
# build it once instead of recomputing jnp.arange(2, n+1) on every call)
_weights = jnp.arange(2, n + 1)


def dixon_price(x):
    return (x[0] - 1) ** 2 + jnp.sum(_weights * (2 * x[1:] ** 2 - x[:-1]) ** 2)


def make_sub_problem(subp):

    start = subp * block  # static python int -> baked into the trace, not a jax value

    # --- pure function of (v, x_fixed) -- no closure over mutable x_init ---
    def sub_obj(v, x_fixed):
        x = jax.lax.dynamic_update_slice(x_fixed, v, (start,))
        return dixon_price(x)

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

        # Use modopt's concrete ProblemLite container, supplying the
        # pre-compiled obj/grad directly instead of JaxProblem (which
        # would re-jit a fresh jax_obj internally on every call).
        problem = mo.ProblemLite(
            x0=v0,
            obj=obj,
            grad=grad,
            xl=-np.inf,
            xu=np.inf,
        )

        optimizer = mo.SLSQP(problem, solver_options={'maxiter': 3000, 'ftol': 1e-7}, turn_off_outputs=True)
        # optimizer = mo.IPOPT(problem, solver_options={'max_iter': 6000, 'tol': 1e-7}, turn_off_outputs=True)

        t1 = time.perf_counter()
        optimizer.solve()
        t2 = time.perf_counter()
        opt_time = t2 - t1
        times.append(opt_time + (times[-1] if len(times) > 0 else 0))

        # optimizer.print_results()

        objective.append(optimizer.results['fun'])

        x_init[subp] = np.asarray(optimizer.results['x'])

        # gc.collect()
        return x_init

    return sub_problem


# run the function generator to generate subproblems
subP_functions = []
for i in range(N):
    subPfunc = make_sub_problem(i)
    subP_functions.append(subPfunc)





# size = int(n / N)
# v_init = []
# for i in range(N): v_init.append(np.ones(size))


# opt = BCD(subproblems=subP_functions,
#             x_init=v_init,
#             solution=[solution],  # known solution for convergence criterion,
#             eps=0.01, # percent solution tolerance
#             )

# tracemalloc.start()

# opt.solve(max_iter=1000)

# print('found solution: ', opt.x)
# print('known solution: ', solution)

# current, peak = tracemalloc.get_traced_memory()
# # print(f"Current: {current / 10**6:.2f} MB")
# print(f"Peak: {peak / 10**6:.2f} MB")

# tracemalloc.stop()

# # print('Solution: ', opt.x)
# # print('Time (s): ', opt.time)
# print('Optimization time (s): ', times[-1])




from scipy.stats.qmc import LatinHypercube, scale

num = 100
sampler = LatinHypercube(d=n, seed=42)
samples = scale(sampler.random(num), l_bounds=-10 * np.ones(n), u_bounds=10 * np.ones(n))

times = []
num_success = 0
for i in range(num):
    print('Run ', i + 1, ' of ', num)

    v_init = np.split(samples[i], N)

    opt = BCD(subproblems=subP_functions,
            x_init=v_init,
            solution=[solution],  # known solution for convergence criterion,
            eps=0.01, # percent solution tolerance
            )

    opt.solve(max_iter=1000)

    # print('Solution: ', opt.x)
    solution = np.concatenate(opt.x)
    print(min(solution), max(solution))
    print('Total time (s): ', opt.tf)

    if opt.success:
        num_success += 1
        times.append(opt.tf)


print('Number of successful runs: ', num_success, ' out of ', num)

mean_time = np.mean(times)
std_time = np.std(times)
print('mean time (s): ', mean_time)
print('std time (s): ', std_time)



np.savez('dixon_price_albcd_time_data_n100_N2_07172026.npz', times=times)