import numpy as np
import matplotlib.pyplot as plt

# Define the nonlinear system
def f(x):
    """Nonlinear vector function f(x) = 0"""
    return np.array([
        np.exp(x[0]) + x[1]**3 - 3,
        x[0]**2 + np.sin(x[1]) - 2
    ])

# Jacobian for Newton's method
def J(x):
    return np.array([
        [np.exp(x[0]), 3*x[1]**2],
        [2*x[0], np.cos(x[1])]
    ])

# Gauss–Seidel-like nonlinear iteration
def gauss_seidel_nonlinear(x0, tol=1e-10, max_iter=200):
    x = x0.copy()
    errors = []
    for k in range(max_iter):
        # Update x[0] from f1 = 0: e^x0 = 3 - y^3
        rhs = 3 - x[1]**3
        if rhs <= 0:
            rhs = 1e-8  # avoid log of nonpositive
        x[0] = np.log(rhs)

        # Update x[1] from f2 = 0: sin(y) = 2 - x^2
        rhs = 2 - x[0]**2
        rhs = np.clip(rhs, -1, 1)  # keep within sine range
        x[1] = np.arcsin(rhs)

        err = np.linalg.norm(f(x))
        errors.append(err)
        if err < tol:
            break
    return x, errors

# Newton's method
def newton_method(x0, tol=1e-10, max_iter=50):
    x = x0.copy()
    errors = []
    for k in range(max_iter):
        fx = f(x)
        Jx = J(x)
        delta = np.linalg.solve(Jx, -fx)
        x = x + delta
        err = np.linalg.norm(f(x))
        errors.append(err)
        if err < tol:
            break
    return x, errors

# Initial guess (moderately far from root)
x0 = np.array([15, 15])

x_gs, err_gs = gauss_seidel_nonlinear(x0)
x_newton, err_newton = newton_method(x0)

print("Gauss–Seidel solution:", x_gs)
print("Newton solution:", x_newton)

# Plot convergence histories
plt.figure(figsize=(8,5))
plt.semilogy(err_gs, 'o-', label='Gauss–Seidel (nonlinear)', linewidth=2)
plt.semilogy(err_newton, 's-', label='Newton’s Method', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Residual Norm ||f(x)|| (log scale)')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()