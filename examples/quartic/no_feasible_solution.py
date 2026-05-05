import numpy as np
import matplotlib.pyplot as plt

# success data
x1_new = np.array([1., 0.375, 0.375, 0.2109375, 0.2109375, 0.11865234, 0.11865234, 0.06674194, 0.06674194, 0.03754234, 0.03754234, 0.02111757, 0.02111757, 0.01187863, 0.01187863, 0.00668173, 0.00668173, 0.00375847, 0.00375847, 0.00211414, 0.00211414, 0.0011892, 0.0011892, 0.00044595, 0.00044595, 0.00016723, 0.00016723, 0.00016723, 0.00016723])
x2_new = np.array([0.5, 0.5, 0.28125, 0.28125, 0.15820312, 0.15820312, 0.08898926, 0.08898926, 0.05005646, 0.05005646, 0.02815676, 0.02815676, 0.01583818, 0.01583818, 0.00890897, 0.00890897, 0.0050113, 0.0050113, 0.00281886, 0.00281886, 0.00158561, 0.00158561, 0.0008919, 0.0008919, 0.00033446, 0.00033446, 0.00012542, 0.00012542, 0.00012542])


plt.rcParams.update({'font.size': 14})

x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 - 1.5 * X * Y
levels = np.linspace(0, max(Z.flatten()), 30)
# plt.contour(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.4, linewidths=0.5)
# plt.contourf(X, Y, Z, levels=levels, cmap='Blues_r', alpha=0.5)
plt.contour(X, Y, Z, levels=levels, cmap='Greens_r', alpha=0.4, linewidths=0.5)
plt.contourf(X, Y, Z, levels=levels, cmap='Greens_r', alpha=0.5)

# plt.plot(x1, x2, 's-', color='tab:red', linewidth=2.5, markersize=6, mec='k', zorder=10)
plt.plot(x1_new, x2_new, 'o-', color='tab:blue', linewidth=2.5, markersize=6, mec='k', zorder=10)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, 2*x, '--', color='black', linewidth=2, alpha=0.5)
plt.plot(x, -2*x, '--', color='black', linewidth=2, alpha=0.5)

plt.fill_between(
    x,
    1.5,        # top of plot
    2*x,        # constraint line
    color='black',
    alpha=0.4,
)

plt.fill_between(
    x,
    -1.5,        # bottom of plot
    -2*x,        # constraint line
    color='black',
    alpha=0.4,
)

# Partial derivatives
dZ_dx1 = 2 * X - 1.5 * Y
dZ_dx2 = 2 * Y - 1.5 * X

# Zero level sets of the derivatives
plt.contour(X, Y, dZ_dx1, levels=[0], colors='tab:purple', linewidths=2, linestyles='dotted', alpha=1)
plt.contour(X, Y, dZ_dx2, levels=[0], colors='tab:olive', linewidths=2, linestyles='dotted', alpha=1)

ticks = [-1, 0, 1]
plt.xticks(ticks)
plt.yticks(ticks)

plt.gca().set_aspect('equal')

plt.savefig('no_feasible_solution_conference_example.pdf', bbox_inches='tight')
plt.show()