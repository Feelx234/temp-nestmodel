from collections import Counter
import networkx as nx

def draw_networkx_causal(G, labels=False):
    """Creates a plot of the causal graph G"""
    include_multiplicity=False
    edge_labels = {}
    if hasattr(G, 'identifiers'):
        edge_multiplicity = Counter(map(tuple, G.edges))
        include_multiplicity = max(edge_multiplicity.values())>1

        pos = {i : G.identifiers[i,:] for i in range(len(G.identifiers))}
    else:
        pos = {i : divmod(i, G.num_nodes_per_time) for i in range(G.num_nodes)}
    G_nx = G.to_nx()
    in_time_edges = []
    between_time_edges = []
    for u,v in G_nx.edges:
        if pos[u][0]==pos[v][0]:
            in_time_edges.append((u,v))
        else:
            between_time_edges.append((u,v))

    if include_multiplicity: # add multiplicity to in between edges
        for edge in between_time_edges:
            multiplicity = edge_multiplicity[edge]
            if multiplicity > 1:
                edge_labels[edge]=str(multiplicity)

    nx.draw_networkx_nodes(G_nx, pos)
    nx.draw_networkx_edges( # From https://stackoverflow.com/questions/52588453/creating-curved-edges-with-networkx-in-python3
        G_nx, pos, edgelist=in_time_edges,
        connectionstyle="arc3,rad=0.2"  # <-- THIS IS IT
    )

    nx.draw_networkx_edges(
        G_nx, pos, edgelist=between_time_edges, edge_color="gray"
    )
    if include_multiplicity:
        nx.draw_networkx_edge_labels(
            G_nx, pos, edge_labels=edge_labels,
        )
    if labels is True:
        nx.draw_networkx_labels(G_nx, pos, font_size=22, font_color="whitesmoke")


def draw_networkx_temp(G_t):
    """Create a plot of the temporal graph G_t"""
    for t, G in enumerate(G_t.slices):
        G_nx = G.to_nx()
        pos = {i : (t, i) for i in G_nx.nodes}

        nx.draw_networkx_nodes(G_nx, pos)
        nx.draw_networkx_edges( # From https://stackoverflow.com/questions/52588453/creating-curved-edges-with-networkx-in-python3
            G_nx, pos,
            connectionstyle="arc3,rad=0.2"  # <-- THIS IS IT
        )
