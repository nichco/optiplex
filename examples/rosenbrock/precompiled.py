from optiplex import BCD
import jax
import jax.numpy as jnp
import modopt as mo
import numpy as np
import warnings
warnings.filterwarnings("ignore")

n = 100  # dimension
N = 10    # number of subproblems

block = n // N  # size of each subproblem's variable block

objective = []

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
        # obj/grad callables directly, instead of JaxProblem (which
        # would re-jit a fresh jax_obj internally on every call)
        problem = mo.ProblemLite(
            x0=v0,
            obj=obj,
            grad=grad,
            xl=-np.inf,
            xu=np.inf,
        )

        optimizer = mo.SLSQP(problem, solver_options={'maxiter': 6000, 'ftol': 1e-7}, turn_off_outputs=True)
        optimizer.solve()
        objective.append(optimizer.results['fun'])
        x_init[subp] = np.asarray(optimizer.results['x'])

        return x_init

    return sub_problem


# run the function generator to generate subproblems
subP_functions = []
for i in range(N):
    subPfunc = make_sub_problem(i)
    subP_functions.append(subPfunc)






"""
size = int(n / N)
guess = np.array([-1.2, 1] * (n // 2))
v_init = []
# for i in range(N): v_init.append(np.zeros(size))
# for i in range(N): v_init.append(np.ones(size) * 3)
for i in range(N): v_init.append(guess[i*size:(i+1)*size])


opt = BCD(subproblems=subP_functions,
          x_init=v_init,
          solution=np.ones(n),  # known solution for convergence criterion,
          eps=0.01, # percent solution tolerance
          )

tracemalloc.start()

opt.solve(max_iter=100)

current, peak = tracemalloc.get_traced_memory()
# print(f"Current: {current / 10**6:.2f} MB")
print(f"Peak: {peak / 10**6:.2f} MB")

tracemalloc.stop()

# print('Solution: ', opt.x)
solution = np.concatenate(opt.x)
print(min(solution), max(solution))
# print('Total time (s): ', opt.time)
print('Optimization time (s): ', times[-1])
"""



from scipy.stats.qmc import LatinHypercube, scale

num = 100#50
sampler = LatinHypercube(d=n, seed=42)
samples = scale(sampler.random(num), l_bounds=-1.5 * np.ones(n), u_bounds=1.5 * np.ones(n))

solution_a = np.ones(n)
solution_b = np.insert(np.ones(n - 1), 0, -1)

times = []
num_success = 0
for i in range(num):
    print('Run ', i + 1, ' of ', num)

    v_init = np.split(samples[i], N)

    opt = BCD(subproblems=subP_functions,
            x_init=v_init,
            # solution=np.ones(n),  # known solution for convergence criterion,
            solution=[solution_a, solution_b],  # known solution for convergence criterion,
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



np.savez('albcd_time_data_n100_N10_07162026.npz', times=times)


# n=100; N=2
# mean_time = 0.78
# std_time = 0.24

# n=200; N=2
# mean_time = 1.16
# std_time = 0.28

# n=400; N=2
# mean_time = 4.89
# std_time = 1.12

# n=600; N=2
# mean_time = 12.81
# std_time = 3.23

# n=800; N=2
# mean_time = 36.04
# std_time = 9.89

# n=1000; N=2
# mean_time = 64.94
# std_time = 19.59







# n=100; N=5
# mean_time = 2.90
# std_time = 0.85

# n=200; N=5
# mean_time = 3.19
# std_time = 1.02

# n=400; N=5
# mean_time = 4.12
# std_time = 1.25

# n=600; N=5
# mean_time = 2.93
# std_time = 0.71

# n=800; N=5
# mean_time = 4.55
# std_time = 0.91

# n=1000; N=5
# mean_time = 9.25
# std_time = 2.39







# n=100; N=10
# mean_time = 5.04
# std_time = 2.19

# n=200; N=10
# mean_time = 5.51
# std_time = 2.15

# n=400; N=10
# mean_time = 5.83
# std_time = 1.80

# n=600; N=10
# mean_time = 
# std_time = 

# n=800; N=10
# mean_time = 
# std_time = 

# n=1000; N=10
# mean_time = 
# std_time = 