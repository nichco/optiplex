"""Run both failure-case example scripts and plot their results side by side."""

import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run_script_as_module(filename):
    """Execute a script as a module and return it, without letting it block on plt.show()."""
    module_name = os.path.splitext(filename)[0]
    path = os.path.join(HERE, filename)

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        spec.loader.exec_module(module)
    finally:
        plt.show = original_show
        plt.close('all')  # discard the figure the script created on its own

    return module


global_mod = run_script_as_module('global_failure_example.py')
infeasible_mod = run_script_as_module('no_feasible_solution.py')

print('Global failure example solution: ', global_mod.opt.x)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 1.5 * X * Y
levels = np.linspace(0, max(Z.flatten()), 30)
dZ_dx1 = 2 * X - 1.5 * Y
dZ_dx2 = 2 * Y - 1.5 * X


def draw_objective_contours(ax):
    ax.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
    ax.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)


def draw_derivative_contours(ax):
    ax.contour(X, Y, dZ_dx1, levels=[0], colors='tab:purple', linewidths=2, linestyles='dotted', alpha=1)
    ax.contour(X, Y, dZ_dx2, levels=[0], colors='tab:olive', linewidths=2, linestyles='dotted', alpha=1)


def format_axes(ax):
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect('equal')


# left: global failure example (single active constraint)
draw_objective_contours(ax1)
ax1.plot(global_mod.x1_history, global_mod.x2_history, 'o-', color='tab:red', linewidth=2.5,
         markersize=6, mec='k', zorder=10)
ax1.plot(x, 2 * x, '--', color='black', linewidth=2, alpha=0.5)
ax1.fill_between(x, 1.5, 2 * x, color='black', alpha=0.4)
draw_derivative_contours(ax1)
format_axes(ax1)

# right: no feasible solution (two active constraints)
draw_objective_contours(ax2)
ax2.plot(infeasible_mod.x1_new, infeasible_mod.x2_new, 'o-', color='tab:blue', linewidth=2.5,
         markersize=6, mec='k', zorder=10)
ax2.plot(x, 2 * x, '--', color='black', linewidth=2, alpha=0.5)
ax2.plot(x, -2 * x, '--', color='black', linewidth=2, alpha=0.5)
ax2.fill_between(x, 1.5, 2 * x, color='black', alpha=0.4)
ax2.fill_between(x, -1.5, -2 * x, color='black', alpha=0.4)
draw_derivative_contours(ax2)
format_axes(ax2)

plt.tight_layout()
plt.savefig('bcd_failure.pdf', bbox_inches='tight')
plt.show()
