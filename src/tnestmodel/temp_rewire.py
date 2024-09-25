import numpy as np
from numba import njit
from numba.typed import Dict  #pylint: disable=no-name-in-module
from tnestmodel.temp_fast_graph import SparseTempFastGraph

class GroupNodeByColor():
    def __init__(self, G_temp, h, s = None):
        self.G_temp = G_temp
        self.h = h
        if s is None:
            self.struct = G_temp.s#get_temporal_wl_struct(h=h, base_edges=True)
        else:
            self.struct = s

        self.last_colors = np.zeros(G_temp.num_nodes, dtype=np.int64) # assumes degree zero nodes have color zero!
        self.nodes_by_color = Dict()
        self.position_for_node = np.arange(G_temp.num_nodes)
        self.size_by_color = Dict()
        self.d = None

    def prepare(self, d):
        self.d = d
        self.struct.reset_colors(d=d, mode="global")
        if self.G_temp.is_directed:
            self.last_colors[:]=0
            self.nodes_by_color.clear()
            self.nodes_by_color[0] = np.arange(len(self.last_colors), dtype=np.int64)
            self.position_for_node = np.arange(len(self.last_colors))
            self.size_by_color[0] = len(self.last_colors)

    def calc_for_slice(self, _, t):
        affected_nodes = self.struct.advance_time(t)
        affected_nodes = np.fromiter(affected_nodes, count = len(affected_nodes), dtype=np.int32)
        update_dict(self.nodes_by_color, self.last_colors, self.struct.current_colors, affected_nodes, self.position_for_node, self.size_by_color)
        return 0

@njit(cache=True)
def update_dict(d, last_colors, new_colors, affected_nodes, position_for_node, size_by_color):
    for v in affected_nodes:
        new_color = new_colors[v]
        last_color = last_colors[v]
        if last_color == new_color:
            continue
        # delete old entry
        arr = d[last_color]
        size = size_by_color[last_color]
        if size==1:
            del d[last_color]
            del size_by_color[last_color]
        else:
            pos = position_for_node[v]
            arr[pos] = arr[size-1]
            position_for_node[arr[pos]] = pos
            size_by_color[last_color]-=1

        # add to new color
        if new_color not in d:
            d[new_color] = np.empty(2, dtype=np.int64)
            size_by_color[new_color]=0
        size = size_by_color[new_color]
        if len(d[new_color]) == size:
            new_size = min(len(last_colors), 2*size)
            assert new_size>size
            tmp = np.empty(new_size, dtype=np.int64)
            tmp[:size] = d[new_color]
            d[new_color] = tmp
        d[new_color][size]=v
        position_for_node[v]=size
        size_by_color[new_color]+=1

        last_colors[v] = new_colors[v]


from nestmodel.fast_rewire import sample_without_replacement, count_in_degree

@njit(cache=True)
def _dir_sample_source_only_direct(edges, nodes_by_class, class_sizes, partition, block):
    """Perform direct sampling of the in-NeSt model"""

    for i in range(len(block)):
        lower = block[i,0]
        upper = block[i,1]
        node1 = edges[lower, 0]
        source_nodes = nodes_by_class[partition[node1]][:class_sizes[partition[node1]]]
        degree_counts = count_in_degree(edges[lower:upper])
        n = 0
        for v, degree in degree_counts.items():
            tmp = sample_without_replacement(source_nodes, degree, avoid=v)
            for u in tmp:
                edges[n,0] = u
                edges[n,1] = v
                n+=1

from nestmodel.fast_rewire import _set_seed

def rewire_temporal_graph_dir(G_temp : SparseTempFastGraph, h=-1, d=-1, seed=None, wl_seed=0):
    assert G_temp.is_directed
    G_temp.assign_colors_to_slices(h, d, seed=wl_seed, sorting_strategy="source")
    grouper = GroupNodeByColor(G_temp, h=h)
    grouper.prepare(d)
    all_edges = []
    if not seed is None:
        _set_seed(seed) # numba seed
        np.random.seed(seed) # numpy seed, seperate from numba seed

    for G, t in zip(G_temp.slices, G_temp.times):
        grouper.calc_for_slice(G, t)
        edges = G.global_edges.copy()
        partition = grouper.last_colors
        blocks = G.block_indices[d]
        _dir_sample_source_only_direct(edges, grouper.nodes_by_color, grouper.size_by_color, partition, blocks)
        all_edges.append(edges)

    return SparseTempFastGraph(all_edges, G_temp.is_directed, G_temp.num_nodes, times=G_temp.times)


