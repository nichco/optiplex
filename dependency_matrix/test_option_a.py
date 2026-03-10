import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

functions = [r'$g_0,h_0$', r'$g_1,h_1$', r'$g_2,h_2$', r'$\vdots$', r'$g_N,h_N$']
variables = [r'$x_0$', r'$x_1$', r'$x_2$', r'$...$', r'$x_N$']

dependency_matrix = np.array([
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 0, 1],
])

plt.figure(figsize=(3, 3))

cmap = ListedColormap(['white', 'steelblue'])

plt.imshow(dependency_matrix, cmap=cmap, vmin=0, vmax=1)

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')

plt.xticks(np.arange(len(variables)), labels=variables)
plt.yticks(np.arange(len(functions)), labels=functions)

# Add grid lines between cells
ax = plt.gca()
ax.set_xticks(np.arange(-0.5, len(variables), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(functions), 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=10)
ax.tick_params(which='minor', bottom=False, left=False)

# Turn off all ticks and spines
ax.tick_params(which='both', bottom=False, left=False, top=False)
for spine in ax.spines.values():
    spine.set_visible(False)

# add vertical line between the first two columns
ax.axvline(x=0.5, color='black', linestyle='-', linewidth=1)
ax.axhline(y=0.5, color='black', linestyle='-', linewidth=1)

plt.tight_layout()
plt.show()