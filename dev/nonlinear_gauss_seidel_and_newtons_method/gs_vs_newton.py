# import numpy as np
# import matplotlib.pyplot as plt

# # Problem setup
# np.random.seed(0)
# n = 20  # moderately large nonlinear system
# A = np.random.rand(n, n)
# A = A + n * np.eye(n)  # diagonally dominant to ensure stability
# b = np.random.rand(n)

# # Nonlinear system definition: f(x) = x^2 + A x - b = 0 (elementwise x^2)
# def f(x):
#     return x**2 + A @ x - b

# # Jacobian of f(x): J(x) = diag(2x) + A
# def J(x):
#     return np.diag(2 * x) + A

# # Gauss-Seidel-like nonlinear iteration:
# # We treat the nonlinear term x_i^2 as part of the local equation
# def gauss_seidel_nonlinear(x0, tol=1e-10, max_iter=200):
#     x = x0.copy()
#     errors = []
#     for k in range(max_iter):
#         x_old = x.copy()
#         for i in range(n):
#             # Compute the nonlinear update for component i
#             sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x_old[i + 1:])
#             # Solve x_i^2 + A_ii * x_i + sigma - b_i = 0
#             a = 1.0
#             b_lin = A[i, i]
#             c = sigma - b[i]
#             # quadratic formula (choose the root closer to old value)
#             disc = b_lin**2 - 4 * a * c
#             if disc < 0:
#                 disc = 0  # avoid complex numbers
#             root1 = (-b_lin + np.sqrt(disc)) / (2 * a)
#             root2 = (-b_lin - np.sqrt(disc)) / (2 * a)
#             x[i] = root1 if abs(root1 - x_old[i]) < abs(root2 - x_old[i]) else root2

#         err = np.linalg.norm(f(x)) / np.linalg.norm(b)
#         errors.append(err)
#         if err < tol:
#             break
#     return x, errors

# # Newton's method
# def newton_method(x0, tol=1e-10, max_iter=50):
#     x = x0.copy()
#     errors = []
#     for k in range(max_iter):
#         fx = f(x)
#         Jx = J(x)
#         delta = np.linalg.solve(Jx, -fx)
#         x = x + delta
#         err = np.linalg.norm(f(x)) / np.linalg.norm(b)
#         errors.append(err)
#         if err < tol:
#             break
#     return x, errors

# # Run both methods
# x0 = np.zeros(n)
# x_gs, err_gs = gauss_seidel_nonlinear(x0)
# x_newton, err_newton = newton_method(x0)

# # Plot convergence histories
# # plt.figure(figsize=(8,5))
# plt.semilogy(err_gs, label='Gauss-Seidel (nonlinear)', linewidth=2)
# plt.semilogy(err_newton, label='Newtons method', linewidth=2)
# plt.xlabel('Iteration')
# plt.ylabel('Residual norm ||f(x)|| / ||b|| (log scale)')
# plt.legend()
# plt.grid(True, which="both", ls="--", lw=0.5)
# plt.tight_layout()
# plt.show()




"""
import numpy as np
import time
import matplotlib.pyplot as plt

# Nonlinear system
def f(x):
    # x = [x0, x1]
    return np.array([
        x[0]**2 + x[1] - 11,
        x[0] + x[1]**2 - 7
    ])

# Jacobian for Newton's method
def J(x):
    return np.array([
        [2*x[0], 1],
        [1, 2*x[1]]
    ])

# Gauss-Seidel-like nonlinear iteration
def gauss_seidel_nonlinear(x0, tol=1e-14, max_iter=1000):
    x = x0.copy()
    errors = []
    for k in range(max_iter):
        x_old = x.copy()
        # Update x[0] using latest y
        x[0] = np.sqrt(max(0, 11 - x[1]))  # one of possible real roots
        # Update x[1] using latest x[0]
        x[1] = np.sqrt(max(0, 7 - x[0]))
        err = np.linalg.norm(f(x))
        errors.append(err)
        if err < tol:
            break
    return x, errors

# Newton's method
def newton_method(x0, tol=1e-14, max_iter=1000):
    x = x0.copy()
    errors = []
    for k in range(max_iter):
        fx = f(x)
        delta = np.linalg.solve(J(x), -fx)
        x = x + delta
        err = np.linalg.norm(f(x))
        errors.append(err)
        if err < tol:
            break
    return x, errors

# Starting guess (not too close to root)
x0 = np.array([600.0, -30000.0])

x_gs, err_gs = gauss_seidel_nonlinear(x0)
x_newton, err_newton = newton_method(x0)

# Plot convergence
plt.figure(figsize=(5,3))
plt.semilogy(err_gs, 'o-', label='Gauss-Seidel', linewidth=2)
plt.semilogy(err_newton, 's-', label='Newton', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.legend()
plt.grid(color='lavender', alpha=0.5)
# plt.tight_layout()
plt.show()
"""



import numpy as np
import time
import matplotlib.pyplot as plt

# Nonlinear system
def f(x):
    return np.array([
        x[0]**2 + x[1] - 11,
        x[0] + x[1]**2 - 7
    ])

# Jacobian for Newton's method
def J(x):
    return np.array([
        [2*x[0], 1],
        [1, 2*x[1]]
    ])

# Gauss-Seidel-like nonlinear iteration
def gauss_seidel_nonlinear(x0, tol=1e-14, max_iter=1000):
    x = x0.copy()
    errors = []
    times = []
    t_start = time.perf_counter()

    for k in range(max_iter):
        x_old = x.copy()
        # Update x[0] and x[1] sequentially
        x[0] = np.sqrt(max(0, 11 - x[1]))
        x[1] = np.sqrt(max(0, 7 - x[0]))
        
        err = np.linalg.norm(f(x))
        errors.append(err)
        times.append(time.perf_counter() - t_start)

        if err < tol:
            break
    return x, errors, times

# Newton's method
def newton_method(x0, tol=1e-14, max_iter=1000):
    x = x0.copy()
    errors = []
    times = []
    t_start = time.perf_counter()

    for k in range(max_iter):
        fx = f(x)
        delta = np.linalg.solve(J(x), -fx)
        x = x + delta
        
        err = np.linalg.norm(f(x))
        errors.append(err)
        times.append(time.perf_counter() - t_start)

        if err < tol:
            break
    return x, errors, times

"""
# Starting guess
x0 = np.array([600.0, -30000.0])

# Run both methods
x_gs, err_gs, t_gs = gauss_seidel_nonlinear(x0)
x_newton, err_newton, t_newton = newton_method(x0)

plt.figure(figsize=(4,2.5))
plt.semilogy(t_gs, err_gs, 'o-', label='Gauss-Seidel', linewidth=2)
plt.semilogy(t_newton, err_newton, 's-', label='Newton', linewidth=2)
plt.xlabel('Wall Time (s)')
plt.ylabel('Residual Norm')
plt.legend()
plt.grid(color='lavender', alpha=0.5)
plt.tight_layout()
plt.show()
"""


plt.figure(figsize=(4,2.5))

n = 100

for i in range(n):
    np.random.seed(i)
    x0 = np.random.uniform(-300, 300, 2)
    # while x0 == 0:
    #     x0 = np.random.uniform(-150, 150)

    x_gs, err_gs, t_gs = gauss_seidel_nonlinear(x0)
    x_newton, err_newton, t_newton = newton_method(x0)

    # plt.semilogy(t_gs, err_gs, 'o-', label='_nolegend_', linewidth=2, color='tab:blue')
    # plt.semilogy(t_newton, err_newton, 's-', label='_nolegend_', linewidth=2, color='tab:orange')
    if i == 0:
        plt.semilogy(t_gs, err_gs, label='Gauss-Seidel', linewidth=2, color='tab:blue', alpha=0.7)
        plt.semilogy(t_newton, err_newton, label='Newton', linewidth=2, color='tab:orange', alpha=0.7)
    else:
        plt.semilogy(t_gs, err_gs, label='_nolegend_', linewidth=2, color='tab:blue', alpha=0.4)
        plt.semilogy(t_newton, err_newton, label='_nolegend_', linewidth=2, color='tab:orange', alpha=0.4)


plt.xlabel('Wall Time (s)')
plt.ylabel('Residual Norm')
plt.legend()
plt.grid(color='lavender', alpha=0.5)
plt.ylim(1e-14, 1e4)

import matplotlib.ticker as ticker
ax = plt.gca()
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)       # always use scientific notation
formatter.set_powerlimits((-2,2))    # limits for when to switch to sci. notation
ax.xaxis.set_major_formatter(formatter)

plt.savefig('gs_vs_newton.pdf', bbox_inches='tight')
plt.show()