
from typing import Tuple
import numpy as np
from nestmodel.fast_graph import FastGraph
from nestmodel.utils import make_directed


class TempFastGraph():
    """A class to model temporal graphs through time slices
    Each timeslices contains all potential nodes (even if they have degree zero)
    """
    def __init__(self, edges, is_directed, num_nodes=None):
        self.num_times=len(edges)
        for edges_t in edges:
            assert edges_t.shape[0]>0
            assert edges_t.shape[1]==2
        self.slices=[FastGraph(v, is_directed=is_directed, num_nodes=num_nodes) for v in edges]
        self.num_nodes = max(G.num_nodes for G in self.slices)
        self.is_directed = is_directed

    def get_causal_completion(self, return_temporal_info=False) -> Tuple[FastGraph, Tuple[int, int]]:
        """ Returns the directed graph that corresponds to this temporal graph
        Returns a normal directed graph in which all nodes have out neighborhoods
        that are identical to their successors in the temporal graph
        """
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


def relabel_edges(edges):
    """relabels nodes such that they start from 0 consecutively"""
    unique = np.unique(edges.ravel())
    mapping = {key:val for key, val in zip(unique, range(len(unique)))}
    unmapping  = unique#{val:key for key, val in mapping.items()}
    out_edges = apply_mapping_to_edges(edges, mapping)
    return out_edges, mapping, unmapping


def apply_mapping_to_edges(edges, mapping):
    """Returns relabeled edges
    The edges are modified such that each node is replaced with its mapped counterpart
    """
    out_edges = np.empty_like(edges)
    for i,(e1,e2) in enumerate(edges):
        out_edges[i,0] = mapping[e1]
        out_edges[i,1] = mapping[e2]
    return out_edges


def to_mapping(values, mapping): #pylint: disable=missing-function-docstring
    return {key : values[val] for key, val in mapping.items()}



class MappedGraph(FastGraph):
    """A FastGraph that removes degree zero nodes through relabeling.s"""
    def __init__(self, edges, is_directed, check_results=False, num_nodes=None):
        self.raw_edges = edges.copy()
        mapped_edges, self.mapping, self.unmapping = relabel_edges(edges)
        self.out_num_nodes = num_nodes
        num_nodes = len(self.unmapping)
        super().__init__(mapped_edges, is_directed, check_results=check_results, num_nodes=num_nodes)

    @property
    def edges(self):
        """Returns edges in their original coordinates"""
        return apply_mapping_to_edges(super().edges, self.unmapping)

    @property
    def internal_edges(self):
        """Returns the edges of the underlying FastGraph object"""
        return super().edges

    @property
    def sparse_out_degree(self):
        """Returns the out degree as a dictionary"""
        return to_mapping(self.out_degree, self.mapping)

    @property
    def sparse_in_degree(self):
        """Returns the in degree as a dictionary"""
        return to_mapping(self.in_degree, self.mapping)


def get_total_degree(l_edges, is_directed, num_nodes):
    """Computes the total degree of a node (i.e. summed over time)"""
    if is_directed:
        in_degree = np.zeros(num_nodes, dtype=np.uint32)
        out_degree = np.zeros(num_nodes, dtype=np.uint32)
        for edges in l_edges:
            for i in range(len(edges)):
                out_degree[edges[i,0]]+=1
                in_degree[edges[i,1]]+=1
        return in_degree, out_degree
    else:
        degree = np.zeros(num_nodes, dtype=np.uint32)
        for edges in l_edges:
            for i in range(len(edges)):
                degree[edges[i,0]]+=1
                degree[edges[i,1]]+=1
        return degree, degree



class SparseTempFastGraph():
    """A class to model temporal graphs through time slices
    Each timeslices contains only those nodes that have degree not zero
    """
    def __init__(self, edges, is_directed, num_nodes=None):
        self.num_times=len(edges)
        for edges_t in edges:
            assert edges_t.shape[0]>0
            assert edges_t.shape[1]==2

        if num_nodes is None:
            num_nodes = max(map(np.max, edges)) + 1
        self.num_nodes = num_nodes
        self.slices=[MappedGraph(v, is_directed=is_directed, num_nodes=num_nodes) for v in edges]
        self.is_directed = is_directed


    def get_sparse_causal_completion(self, return_temporal_info=False) -> Tuple[FastGraph, Tuple[int, int]]:
        """ Returns the directed graph that corresponds to this temporal graph
            Returns a normal directed graph in which all nodes have out neighborhoods
            that are identical to their successors in the temporal graph. This graph is smaller
            because nodes that are guaranteed to be structurally identical were removed.
            The identity of node i of G is stored in G.indentifiers[i] = (t, u)
        """
        _, out_degree = get_total_degree([G.raw_edges for G in self.slices], self.is_directed, self.num_nodes)
        list_successors = [np.empty(out_deg, dtype=np.int32) for out_deg in out_degree]
        num_successors = np.zeros(self.num_nodes, dtype=np.int32)
        num_prev_nodes = np.cumsum([0]+[G.num_nodes for G in self.slices])
        all_edges = []
        T = len(self.slices)
        node_identifier = np.empty((num_prev_nodes[-1],2), dtype=np.int32)
        for t in range(T-1, -1, -1):
            g = self.slices[t]
            local_edges = g.internal_edges
            if not self.is_directed:
                local_edges = make_directed(local_edges)

            need_processing = set()
            for i, j in local_edges:
                i_glob = g.unmapping[i]
                list_successors[i_glob][len(list_successors[i_glob])-num_successors[i_glob]-1] = j + num_prev_nodes[t]
                num_successors[i_glob]+=1
                need_processing.add(i)
            total_edges = sum(num_successors[g.unmapping[i]] for i in need_processing)

            edges = np.empty((total_edges,2), dtype=np.uint32)
            n=0
            for i in need_processing:
                i_glob = g.unmapping[i]
                l = num_successors[i_glob]
                global_name = i + num_prev_nodes[t]
                edges[n:n+l,0] = global_name
                edges[n:n+l,1] = list_successors[i_glob][len(list_successors[i_glob])-l:]
                n+=l
            all_edges.append(edges)

            for i in range(g.num_nodes):# set node identifiers for nodes this slice
                global_name = i + num_prev_nodes[t]
                node_identifier[global_name,0] = t
                node_identifier[global_name,1] = g.unmapping[i]

        all_edges = np.vstack(list(reversed(all_edges)))
        G = FastGraph(all_edges, is_directed=True, num_nodes=num_prev_nodes[-1])
        G.identifiers = node_identifier
        if return_temporal_info:
            return G, (self.num_nodes, T)
        else:
            return G
