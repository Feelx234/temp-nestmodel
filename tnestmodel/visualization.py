import networkx as nx

def draw_networkx_causal(G, num_nodes, ):
    """Creates a plot of the causal graph G"""
    if hasattr(G, 'identifiers'):
        pos = {i : G.identifiers[i,:] for i in range(len(G.identifiers))}
        print(pos, G.to_nx().nodes)
    else:
        pos = {i : divmod(i, num_nodes) for i in range(G.num_nodes)}
    G_nx = G.to_nx()
    in_time_edges = []
    between_time_edges = []
    for u,v in G_nx.edges:
        if pos[u][0]==pos[v][0]:
            in_time_edges.append((u,v))
        else:
            between_time_edges.append((u,v))

    nx.draw_networkx_nodes(G_nx, pos)
    nx.draw_networkx_edges(
        G_nx, pos, edgelist=in_time_edges,
        connectionstyle="arc3,rad=0.2"  # <-- THIS IS IT
    )
    nx.draw_networkx_edges(
        G_nx, pos, edgelist=between_time_edges, edge_color="gray"
    )


def draw_networkx_temp(G_t):
    """Create a plot of the temporal graph G_t"""
    for t, G in enumerate(G_t.slices):
        G_nx = G.to_nx()
        pos = {i : (t, i) for i in G_nx.nodes}

        print(G_nx.nodes())
        print(pos)
        nx.draw_networkx_nodes(G_nx, pos)
        nx.draw_networkx_edges(
            G_nx, pos,
            connectionstyle="arc3,rad=0.2"  # <-- THIS IS IT
        )
