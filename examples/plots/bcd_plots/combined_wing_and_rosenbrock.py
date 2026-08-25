"""Run both BCD example scripts and plot their results side by side."""

import importlib.util
import os
import sys

import matplotlib.pyplot as plt

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


wing_mod = run_script_as_module('distributed_wing_design_jax.py')
rosenbrock_mod = run_script_as_module('run_distributed_2d_rosenbrock.py')

print('Wing design solution: ', wing_mod.opt.x)
print('Rosenbrock solution: ', rosenbrock_mod.opt.x)


plt.rcParams.update({'font.size': 14})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))


def draw_derivative_contours(ax, X, Y, dZ_dx, dZ_dy):
    ax.contour(X, Y, dZ_dx, levels=[0], colors='tab:purple', linewidths=2, linestyles='-.', alpha=1)
    ax.contour(X, Y, dZ_dy, levels=[0], colors='tab:olive', linewidths=2, linestyles='-.', alpha=1)


# left: distributed 2D Rosenbrock
ax1.contour(rosenbrock_mod.X, rosenbrock_mod.Y, rosenbrock_mod.Z, levels=rosenbrock_mod.levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
ax1.contourf(rosenbrock_mod.X, rosenbrock_mod.Y, rosenbrock_mod.Z, levels=rosenbrock_mod.levels, cmap='Blues_r', alpha=0.5)
draw_derivative_contours(ax1, rosenbrock_mod.X, rosenbrock_mod.Y, rosenbrock_mod.dZ_dx1, rosenbrock_mod.dZ_dx2)
ax1.plot(rosenbrock_mod.x1_history, rosenbrock_mod.x2_history, '-o', mec='k', color='tab:red', linewidth=2.5, markersize=7, zorder=10)
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xticks([-1, 0, 1])
ax1.set_yticks([-1, 0, 1])

ax1.set_aspect('equal')

# right: distributed wing design
ax2.contour(wing_mod.B, wing_mod.C, wing_mod.Z, levels=wing_mod.levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
ax2.contourf(wing_mod.B, wing_mod.C, wing_mod.Z, levels=wing_mod.levels, cmap='Blues_r', alpha=0.5)
draw_derivative_contours(ax2, wing_mod.B, wing_mod.C, wing_mod.dZ_db, wing_mod.dZ_dc)
ax2.plot(wing_mod.b_history, wing_mod.c_history, '-o', mec='k', color='tab:red', linewidth=2.5, markersize=7, zorder=10)
ax2.set_xlim(5, 35)
ax2.set_ylim(0.3, 1.5)
ax2.set_xlabel('Wing span')
ax2.set_ylabel('Chord')

ax2.set_box_aspect(1)

plt.tight_layout()
plt.savefig('bcd_results.pdf', bbox_inches='tight')
plt.show()
