from collections import Counter
import numpy as np
import networkx as nx

from matplotlib import colors as mcolors

# the color names thing is from
#  https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
named_colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
fixed_colors = ["red", "yellow", "green", "blue", "deeppink", "aqua", "teal",
                "saddlebrown", "darkviolet", "lawngreen", "deepskyblue"]
remove_colors = ['w', 'k', 'ghostwhite', 'azure', 'honeydew',
                     'snow', 'white', 'seashell', 'ivory', 'aliceblue',
                    'lavenderblush', 'mintcream', 'lavender', 'whitesmoke', 'floralwhite']
for remove_color in remove_colors+fixed_colors:
    del named_colors[remove_color]

by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in named_colors.items())

# "#1f78b4" is the networkx color
sorted_names = [name for hsv, name in by_hsv]
sorted_names = np.array(sorted_names).reshape(13,10).T.ravel()

def to_color(v):
    first_colors = ["#1f78b4"]+fixed_colors
    if v < len(first_colors):
        return first_colors[v]
    v-=len(first_colors)
    assert v<len(sorted_names), f"May only have color value in [0:{len(sorted_names)+len(first_colors)-1}]. You have requested {v+len(first_colors)}"
    return sorted_names[v]

def draw_networkx_causal(G, labels=False, colors=None):
    """Creates a plot of the causal graph G"""
    include_multiplicity=False
    edge_labels = {}
    if hasattr(G, 'identifiers'):
        edge_multiplicity = Counter(map(tuple, G.edges))
        include_multiplicity = max(edge_multiplicity.values())>1

        pos = {i : (G.identifiers[i,1],G.identifiers[i,0],) for i in range(len(G.identifiers))}
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
    if colors is None:
        nx.draw_networkx_nodes(G_nx, pos)
    elif isinstance(colors, int):
        G_rev = G.switch_directions()
        node_wl = G_rev.calc_wl()[colors].ravel()
        node_color = [to_color(wl) for wl in node_wl]
        nx.draw_networkx_nodes(G_nx, pos, node_color=node_color)
    else:
        raise NotImplementedError()
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


def draw_networkx_temp(G_t, colors=None):
    """Create a plot of the temporal graph G_t"""
    if isinstance(colors, int):
        cols = G_t.calc_wl()
        G_t.apply_wl_colors_to_slices(cols)

    for t, G in zip(G_t.times, G_t.slices):
        G_nx = nx.relabel_nodes(G.to_nx(), {i:x for i, x in enumerate(G.unmapping)})
        pos = {i : (t, i) for i in G_nx.nodes}


        if colors is None:
            nx.draw_networkx_nodes(G_nx, pos)
        elif isinstance(colors, int):
            node_wl = G.base_partitions[colors]
            node_color = [to_color(wl) for wl in node_wl]
            nx.draw_networkx_nodes(G_nx, pos, node_color=node_color)
        elif isinstance(colors, (np.ndarray, list, tuple)):
            node_color = [to_color(wl) for wl in colors[t]]
            nx.draw_networkx_nodes(G_nx, pos, node_color=node_color)
        else:
            raise NotImplementedError()
        nx.draw_networkx_edges( # From https://stackoverflow.com/questions/52588453/creating-curved-edges-with-networkx-in-python3
            G_nx, pos,
            arrows=True,
            connectionstyle="arc3,rad=0.2"  # <-- THIS IS IT
        )
