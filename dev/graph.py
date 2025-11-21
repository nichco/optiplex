import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

def hessian(x):
    """
    Compute the Hessian of the n-dimensional Rosenbrock function at point x.
    f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    """
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n - 1):
        # Diagonal element H[i, i]
        H[i, i] += 1200 * x[i]**2 - 400 * x[i+1] + 2
        # Coupling with next variable
        H[i, i+1] += -400 * x[i]
        H[i+1, i] += -400 * x[i]
        # Contribution to next diagonal
        H[i + 1, i + 1] += 200

    return H


# Example usage
n = 60
# x0 = np.ones(n) * -1
# x0 = np.array([-1.2, 1] * (n // 2))
x0 = np.random.uniform(-2, 2, n)
H = hessian(x0)

# Build graph
G = nx.Graph()

# Add nodes
for i in range(n):
    G.add_node(i, label=f"x_{i+1}")

for i in range(n):
    for j in range(i + 1, n):
        weight = abs(H[i, j])  # will be zero if Hessian entry is zero
        G.add_edge(i, j, weight=weight)



# if layout == 'shell': pos = nx.shell_layout(G)
#     elif layout == 'circular': pos = nx.circular_layout(G)
#     elif layout == 'random': pos = nx.random_layout(G)
#     elif layout == 'spring': pos = nx.spring_layout(G)
#     elif layout == 'spectral': pos = nx.spectral_layout(G)
#     elif layout == 'kamada_kawai': pos = nx.kamada_kawai_layout(G)
#     elif layout == 'fruchterman_reingold': pos = nx.fruchterman_reingold_layout(G)
#     elif layout == 'spiral': pos = nx.spiral_layout(G)

# pos = nx.spring_layout(G, seed=0)
pos = nx.random_layout(G)
# pos = nx.shell_layout(G)
# pos = nx.circular_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.spiral_layout(G)
edge_weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    width=1,
)

# nx.draw_networkx_edge_labels(
#     G,
#     pos,
#     edge_labels={(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()},
#     font_color="red",
#     font_size=8,
# )

plt.show()






adjacency_matrix = nx.adjacency_matrix(G).toarray()

clustering = SpectralClustering(
    n_clusters=2,
    affinity="precomputed",
    # assign_labels="kmeans",
    assign_labels="discretize",
    random_state=42
).fit(adjacency_matrix)

labels = clustering.labels_  # cluster assignment for each node


# pos = nx.spring_layout(G, seed=42)
pos = nx.random_layout(G)

colors = ["tab:blue" if labels[i] == 0 else "tab:orange" for i in range(n)]
edge_widths = [G[u][v]["weight"] for u, v in G.edges()]

plt.figure(figsize=(8, 6))
nx.draw(
    G, pos,
    with_labels=True,
    node_color=colors,
    # node_size=900,
    width=1,
    font_size=12
)

plt.show()



