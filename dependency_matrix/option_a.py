import numpy as np
import matplotlib.pyplot as plt

# Example data
functions = [r'$g_0,h_0$', r'$g_1,h_1$', r'$g_2,h_2$', r'$\vdots$', r'$g_N,h_N$']
variables = [r'$x_0$', r'$x_1$', r'$x_2$', r'$...$', r'$x_N$']

dependency_matrix = np.array([
    [1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
])

# fig, ax = plt.subplots()

plt.figure(figsize=(3, 3))

plt.imshow(dependency_matrix, cmap="gray_r", interpolation="nearest")

# Major ticks
plt.xticks(np.arange(len(variables)))
plt.yticks(np.arange(len(functions)))

plt.xticks(np.arange(len(variables)), labels=variables)
plt.yticks(np.arange(len(functions)), labels=functions)

# move x axis labels to the top of the plot
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
# plt.setp(ax.get_xticklabels(), rotation=45, ha="left")

# Create minor ticks at cell boundaries
plt.xticks(np.arange(-0.5, len(variables), 1), minor=True)
plt.yticks(np.arange(-0.5, len(functions), 1), minor=True)

# Draw custom gridlines (cell boundaries)
plt.grid(
    which="minor",
    color='white',
    linestyle='-',
    linewidth=4
)

# ax.tick_params(which="minor", bottom=False, left=False)

# Remove outer spines for clean matrix look (optional)
for spine in plt.gca().spines.values():
    spine.set_visible(False)


# set equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()