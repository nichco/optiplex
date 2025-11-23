import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

def hessian(x):

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
n = 6
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




plt.figure(figsize=(4, 4))


# pos = nx.spring_layout(G, seed=0)
# pos = nx.random_layout(G, seed=1)
pos = nx.shell_layout(G)
# pos = nx.circular_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.spiral_layout(G)
edge_weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="white",
    width=2,
    font_size=12,
    edge_color='black',
    edgecolors="black",
)

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels={(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()},
    font_color="black",
    font_size=10,
)

plt.savefig('rosenbrock_hessian_graph.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()






adjacency_matrix = nx.adjacency_matrix(G).toarray()

clustering = SpectralClustering(
    n_clusters=3,
    affinity="precomputed",
    # assign_labels="kmeans",
    assign_labels="discretize",
    random_state=42
).fit(adjacency_matrix)

labels = clustering.labels_  # cluster assignment for each node
print("Cluster labels for each node:", labels)


# colors = ["tab:blue" if labels[i] == 0 else "tab:orange" for i in range(n)]
colors = ["tab:blue" if labels[i] == 0 else ("tab:orange" if labels[i] == 1 else "tab:green") for i in range(n)]
edge_widths = [G[u][v]["weight"] for u, v in G.edges()]

# plt.figure(figsize=(8, 6))
plt.figure(figsize=(4, 4))
nx.draw(
    G, pos,
    with_labels=True,
    node_color=colors,
    # node_size=900,
    width=1,
    font_size=10,
    edge_color='black',
    edgecolors="black",
)

plt.savefig('rosenbrock_clusters.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()



