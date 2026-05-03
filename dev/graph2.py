import networkx as nx
import matplotlib.pyplot as plt

G = nx.MultiDiGraph()

edges = [
    ("D1", "D2"), ("D2", "D1"),
    ("D2", "D3"), ("D3", "D2"),
    ("D3", "D1"), ("D1", "D3"),
    ("D1", "Global Constraints"), ("D2", "Global Constraints"), ("D3", "Global Constraints"),
    ("D1", "Objective"),          ("D2", "Objective"),          ("D3", "Objective"),
    ("Global Constraints", "Objective"),
]

G.add_edges_from(edges)

colors = {"D1": "#BAD4EA", "D2": "#BAD4EA", "D3": "#BAD4EA",
          "Global Constraints": "#d3d3d3", "Objective": "#d3d3d3"}

# pos = nx.spring_layout(G, seed=0)
# pos = nx.random_layout(G, seed=0)
# pos = nx.shell_layout(G, seed=0)
# pos = nx.circular_layout(G)
# pos = nx.spectral_layout(G)
# pos = nx.spiral_layout(G)

pos = {
    "D1": (-0.1,  0.5),
    "D2": ( 0.0,  1.5),
    "D3": ( 0.1,  0.5),
    "Global Constraints": (-0.1, -0.5),
    "Objective":          ( 0, -1.5),
}

fig, ax = plt.subplots(figsize=(5, 5))
nx.draw_networkx_nodes(G, pos, node_color=[colors[n] for n in G], node_size=2500, ax=ax, edgecolors='black')
labels = {n: n.replace(" ", "\n", 1) if n == "Global Constraints" else n for n in G}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)
nx.draw_networkx_edges(G, pos, connectionstyle="arc3,rad=0.2", arrowsize=20,
                       edge_color="#444", width=1.6, ax=ax,
                       min_source_margin=30, min_target_margin=30)

ax.axis("off")
plt.tight_layout()
plt.savefig("dev/graph.png", dpi=300, transparent=True)
plt.show()