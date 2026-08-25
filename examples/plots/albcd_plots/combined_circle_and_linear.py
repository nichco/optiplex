"""Run both ALBCD example scripts and plot their results side by side."""

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


circle_mod = run_script_as_module('quadratic_albcd_circle_constraint.py')
linear_mod = run_script_as_module('quadratic_albcd_linear_constraint.py')

print('Circle constraint solution: ', circle_mod.opt.x)
print('Linear constraint solution: ', linear_mod.opt.x)


# plt.rcParams.update({'font.size': 14})
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))


def draw_objective_contours(ax):
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2 - 1.5 * X * Y
    levels = np.linspace(0, max(Z.flatten()), 30)
    ax.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
    ax.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)
    return x


def draw_history(ax, mod):
    ax.plot(mod.x1_1_history, mod.x2_2_history, 's-', color='tab:orange', linewidth=2.5,
            markersize=6, zorder=10, mec='k', label=r'$(x_1, y_2)$')
    ax.plot(mod.x1_2_history, mod.x2_1_history, 'o-', color='tab:purple', linewidth=2.5,
            markersize=6, zorder=10, mec='k', label=r'$(x_2, y_1)$')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.legend()
    ax.set_aspect('equal')


# left: linear constraint
x = draw_objective_contours(ax1)
draw_history(ax1, linear_mod)
ax1.plot(x, 2 * x, '--', color='black', linewidth=2, alpha=0.5)
ax1.fill_between(x, 1.5, 2 * x, color='black', alpha=0.4)

# right: circle constraint
draw_objective_contours(ax2)
draw_history(ax2, circle_mod)
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = 0.5 * np.cos(theta)
circle_y = 0.5 * np.sin(theta)
ax2.plot(circle_x, circle_y, '--', color='black', linewidth=2, alpha=0.5)
ax2.fill(circle_x, circle_y, color='black', alpha=0.3)

plt.tight_layout()
plt.savefig('albcd_result.pdf', bbox_inches='tight')
plt.show()
