
from typing import Tuple
import numpy as np
from nestmodel.fast_graph import FastGraph
from nestmodel.utils import make_directed


class TempFastGraph():
    """A class to model temporal graphs through time slices
    Each timeslices contains all potential nodes (even if they have degree zero)
    """
    def __init__(self, edges, is_directed, r=None, num_nodes=None):
        self.num_times=len(edges)
        for edges_t in edges:
            assert edges_t.shape[0]>0
            assert edges_t.shape[1]==2

        if num_nodes is None:
            num_nodes = max(map(np.max, edges)) + 1
        self.slices=[FastGraph(v, is_directed=is_directed, num_nodes=num_nodes) for v in edges]
        self.num_nodes = max(G.num_nodes for G in self.slices)
        self.is_directed = is_directed
        if r is None:
            r = len(self.slices)
        self.r = r

    def get_causal_completion(self) -> FastGraph:
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
        G.num_nodes_per_time = self.num_nodes
        return G

    def calc_wl(self, initial_colors=None, max_depth=None):
        """Compute the WL colors of this graph using the provided initial colors"""
        G_causal_rev = self.get_causal_completion().switch_directions()
        out = G_causal_rev.calc_wl(initial_colors=initial_colors, max_depth=max_depth)
        wl_colors = [colors.reshape((len(self.slices), self.num_nodes)) for colors in out]
        return wl_colors # each row is one timeslice

    def apply_wl_colors_to_slices(self, wl_colors):
        wl_colors_by_slice = [[] for _ in range(len(self.slices))]
        for colors_per_depth in wl_colors:
            for i, slice_colors in enumerate(wl_colors_by_slice):
                slice_colors.append(colors_per_depth[i,:].ravel())
        for G, colors in zip(self.slices, wl_colors_by_slice):
            G.base_partitions = np.array(colors, dtype=np.uint32)
            G.wl_iterations = len(G.base_partitions)
            G.reset_edges_ordered()



    def get_restless_causal_completion(self):
        """Returns a restless causal completion of the temporal graph
        This uses the parameter r to specify the maximum length of restless walks
        """
        all_edges = []
        T = len(self.slices)
        in_degrees = [G.in_degree for G in self.slices]
        out_degrees = [G.out_degree for G in self.slices]
        mapping = [np.arange(self.num_nodes)] * T
        _, max_out_degree = get_rolling_max_degree((in_degrees, out_degrees), mapping, self.is_directed, self.r, self.num_nodes)
        list_successors = [-np.ones(out_deg, dtype=np.int32) for out_deg in max_out_degree]
        time_successors = [np.empty(out_deg, dtype=np.int32) for out_deg in max_out_degree]
        first_index = np.zeros(self.num_nodes, dtype=np.uint32)
        last_index = np.zeros(self.num_nodes, dtype=np.uint32)
        for t in range(T-1, -1, -1):
            g = self.slices[t]

            num_total_edges = len(g.edges) + np.sum(last_index - first_index)

            for u, v in g.edges:
                buff_size = len(list_successors[u])
                u_successors = list_successors[u]
                u_successors[last_index[u] % buff_size] = v + t*self.num_nodes
                time_successors[u][last_index[u] % buff_size] = t
                last_index[u] +=1
            new_edges = np.empty((num_total_edges, 2), dtype=np.uint32)
            n = 0
            for v in range(self.num_nodes):

                buff_size = len(list_successors[v])
                if buff_size ==0:
                    continue
                while (time_successors[v][first_index[v] % buff_size]  > t+self.r) and (first_index[v] < last_index[v]):
                    first_index[v] += 1
                delta = last_index[v]-first_index[v]
                for j in range(0, delta):
                    new_edges[n+j,0] = v + t*self.num_nodes
                    assert list_successors[v][(first_index[v] + j) % buff_size]>=0
                    new_edges[n+j,1] = list_successors[v][(first_index[v] + j) % buff_size]
                n+=delta
            all_edges.append(new_edges[:n, :])
        all_edges = np.vstack(all_edges)
        G = FastGraph(all_edges, is_directed=True, num_nodes=T*self.num_nodes)
        G.num_nodes_per_time = self.num_nodes
        return G



    def sparse_causal_adjacency(self):
        """returns the sparse causal adjacency withour computing the causal graph first"""
        import scipy.sparse as sparse # pylint: disable=import-outside-toplevel
        from itertools import repeat # pylint: disable=import-outside-toplevel
        T = len(self.slices)
        adjacencies = [G.to_coo() for G in self.slices]
        empty_matrix = sparse.coo_matrix(([], ([], [])), shape=adjacencies[0].shape)
        columns = [sparse.vstack(tuple(np.repeat(A,i+1)) + tuple(repeat(empty_matrix, T-i-1))) for i, A in enumerate(adjacencies)]
        return sparse.hstack(columns)


    def reverse_slice_direction(self):
        """Returns a new graph with all edge directions reversed within time slices"""
        reversed_edges = [G.switch_directions().edges for G in self.slices]
        return TempFastGraph(reversed_edges, is_directed=self.is_directed, num_nodes=self.num_nodes)

    def reverse_time(self):
        """Returns a new graph with time reversed"""
        reversed_edges = [G.edges for G in reversed(self.slices)]
        return TempFastGraph(reversed_edges, is_directed=self.is_directed, num_nodes=self.num_nodes)



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


    def to_coo(self):
        """Returns a sparse coo-matrix representation of the graph"""
        from scipy.sparse import coo_matrix # pylint: disable=import-outside-toplevel
        edges = self.edges
        if not self.is_directed:
            edges = make_directed(edges)

        return coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape = (self.out_num_nodes, self.out_num_nodes))


def _get_rolling_max_degree(l_degrees, l_mapping, r, num_nodes):

    # r = 1 is that we we look at the current and the next slice
    # thus need to increase r by 1 to get buf size
    buff_size = r+1
    T = len(l_degrees)
    max_degree = np.zeros(num_nodes, dtype=np.uint32)
    curr_degree = np.zeros(num_nodes, dtype=np.uint32)
    roll_degree = np.zeros((num_nodes, buff_size), dtype=np.uint32)
    roll_time = np.zeros((num_nodes, buff_size), dtype=np.uint32)
    first_index = np.zeros(num_nodes, dtype=np.uint32)
    last_index = np.zeros(num_nodes, dtype=np.uint32)
    for t in range(T):
        mapping = l_mapping[t]
        degrees = l_degrees[t]
        #print(max_degree)
        #print(curr_degree)
        #print(roll_degree)
        #print(first_index)
        #print(last_index)
        #print()
        for v, deg in zip(mapping, degrees):
            curr_degree[v] += deg
            while (roll_time[v, first_index[v] % buff_size]  < t-r) and (first_index[v] < last_index[v]):
                curr_degree[v] -= roll_degree[v, first_index[v] % buff_size]
                first_index[v] += 1
            max_degree[v] = max(max_degree[v], curr_degree[v])
            roll_degree[v, last_index[v] % buff_size] = deg
            last_index[v] += 1
    return max_degree


def get_rolling_max_degree(l_degrees, l_mapping, is_directed, r, num_nodes):
    """Computes the rolling total degree of a node (i.e. summed over time)"""
    assert len(l_degrees[0])==len(l_mapping)
    assert r >= 0
    if is_directed:
        max_in_degree = _get_rolling_max_degree(l_degrees[0], l_mapping, r, num_nodes)
        max_out_degree = _get_rolling_max_degree(l_degrees[1], l_mapping, r, num_nodes)
        return max_in_degree, max_out_degree
    else:
        max_degree = _get_rolling_max_degree(l_degrees[0], l_mapping, r, num_nodes)
        return max_degree, max_degree


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





def get_visible_nodes_per_time(slices, num_nodes):
    """

    For a node to be visible you have the following decisions
    out-deg_=
        >0  => appears
        else, in_deg_=
            =0 => hidden
            else, ind_deg_>
                = 0 => appears
                > 0 => hidden
    """
    active_nodes_list = [None for _ in range(len(slices))]
    num_active_nodes = np.zeros(len(slices), dtype=np.int32)
    in_deg_greater = np.zeros(num_nodes, dtype=np.int32)

    for t, G in reversed(list(enumerate(slices))):
        active_nodes_this_slice = np.zeros(G.num_nodes, dtype=np.bool_)
        for v in range(G.num_nodes):
            v_global = G.unmapping[v]
            if G.out_degree[v] > 0:
                active_nodes_this_slice[v] = 1
                num_active_nodes[t] +=1
            else:
                if G.in_degree[v] > 0 and in_deg_greater[v_global]==0:
                    active_nodes_this_slice[v] = 1
                    num_active_nodes[t] +=1
            in_deg_greater[v_global]+=G.in_degree[v]
        active_nodes_list[t] = active_nodes_this_slice
    return active_nodes_list, num_active_nodes



class SparseTempFastGraph():
    """A class to model temporal graphs through time slices
    Each timeslices contains only those nodes that have non-zero degree
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


    def reverse_slice_direction(self):
        """Returns a new graph with all edge directions reversed within time slices"""
        reversed_edges = [G.switch_directions().edges for G in self.slices]
        return SparseTempFastGraph(reversed_edges, is_directed=self.is_directed, num_nodes=self.num_nodes)

    def reverse_time(self):
        """Returns a new graph with time reversed"""
        reversed_edges = [G.edges for G in reversed(self.slices)]
        return SparseTempFastGraph(reversed_edges, is_directed=self.is_directed, num_nodes=self.num_nodes)


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

        is_active, vis_nodes = get_visible_nodes_per_time(self.slices, self.num_nodes)
        num_prev_nodes = np.cumsum([0]+list(vis_nodes))
        #num_prev_nodes = np.cumsum([0]+[G.num_nodes for G in self.slices])
        last_of_its_kind = np.full(self.num_nodes, -1, dtype=np.int32)
        all_edges = []
        T = len(self.slices)
        #print(num_prev_nodes)
        node_identifier = np.empty((num_prev_nodes[-1],2), dtype=np.int32)
        for t in range(T-1, -1, -1):
            g = self.slices[t]
            local_edges = g.internal_edges
            if not self.is_directed:
                local_edges = make_directed(local_edges)

            n_temp = 0
            for i in range(g.num_nodes):
                if is_active[t][i]:
                    i_glob = g.unmapping[i]
                    global_name = n_temp + num_prev_nodes[t]
                    last_of_its_kind[i_glob] = global_name
                    n_temp +=1

            #print(local_edges)
            for i, j in local_edges: # i,j are names specific to the timeslice
                i_glob = g.unmapping[i] # name of node i in the global namespace
                j_glob = g.unmapping[j] # name of node j in the global namespace
                assert last_of_its_kind[j_glob] > -1
                #if last_of_its_kind[j_glob] ==-1:
                #    last_of_its_kind[j_glob] = j + num_prev_nodes[t]
                list_successors[i_glob][len(list_successors[i_glob])-num_successors[i_glob]-1] = last_of_its_kind[j_glob]
                num_successors[i_glob]+=1
            total_edges = sum(num_successors[g.unmapping[i]] for i in range(g.num_nodes) if is_active[t][i])

            edges = np.empty((total_edges,2), dtype=np.uint32)
            #print(edges)
            #print(is_active)
            n=0
            n_temp = 0
            for i in range(g.num_nodes):
                if is_active[t][i]:
                    i_glob = g.unmapping[i]
                    l = num_successors[i_glob]
                    global_name = n_temp + num_prev_nodes[t]
                    edges[n:n+l,0] = global_name
                    edges[n:n+l,1] = list_successors[i_glob][len(list_successors[i_glob])-l:]
                    n+=l
                    n_temp +=1
            #print(edges)
            #print()
            #print()
            all_edges.append(edges)

            n_temp = 0
            for i in range(g.num_nodes):# set node identifiers for nodes this slice
                if is_active[t][i]:
                    global_name = n_temp + num_prev_nodes[t]
                    node_identifier[global_name,0] = t
                    node_identifier[global_name,1] = g.unmapping[i]
                    n_temp+=1

        all_edges = np.vstack(list(reversed(all_edges)))
        G = FastGraph(all_edges, is_directed=True, num_nodes=num_prev_nodes[-1])
        G.identifiers = node_identifier
        if return_temporal_info:
            return G, (self.num_nodes, T)
        else:
            return G