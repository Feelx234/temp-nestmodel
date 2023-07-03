from typing import Tuple
import numpy as np
from nestmodel.fast_graph import FastGraph
from nestmodel.utils import make_directed


class TempFastGraph():
    def __init__(self, edges, is_directed, num_nodes=None):
        self.num_times=len(edges)
        for edges_t in edges:
            assert edges_t.shape[0]>0
            assert edges_t.shape[1]==2
        self.slices=[FastGraph(v, is_directed=is_directed, num_nodes=num_nodes) for v in edges]
        self.num_nodes = max(G.num_nodes for G in self.slices)
        self.is_directed = is_directed

    def get_causal_completion(self, return_temporal_info=False) -> Tuple[FastGraph, Tuple[int, int]]:
        all_edges = []
        T = len(self.slices)
        edges = np.empty((0,2), dtype=self.slices[-1].edges.dtype)
        for t in range(T-1, -1, -1):
            g = self.slices[t]
            tmp_edges = g.edges.copy()
            if not self.is_directed:
                tmp_edges = make_directed(tmp_edges)
            tmp_edges[:,1]+=t*self.num_nodes
            edges = np.vstack((tmp_edges, edges))

            new_edges = edges.copy()
            new_edges[:,0]+=t*self.num_nodes
            all_edges.append(new_edges)
        all_edges = np.vstack(list(reversed(all_edges)))
        G = FastGraph(all_edges, is_directed=True, num_nodes=T*self.num_nodes)
        if return_temporal_info:
            return G, (self.num_nodes, T)
        else:
            return G
