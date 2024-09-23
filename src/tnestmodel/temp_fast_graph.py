
from typing import Tuple
from itertools import product
import warnings
import numpy as np
from nestmodel.fast_graph import FastGraph
from nestmodel.utils import make_directed, switch_in_out
from tnestmodel.temp_utils import partition_temporal_edges
from tnestmodel.temp_wl import TemporalColorsStruct, _compute_d_rounds
from tnestmodel.temp_utils import temp_undir_to_directed
from tnestmodel.causal_completion import get_dense_identifiers, get_edges_dense_causal_completion





class TempFastGraph():
    """A class to model temporal graphs through time slices
    Each timeslices contains all potential nodes (even if they have degree zero)
    """
    def __init__(self, slices, is_directed, times=None, r=None, num_nodes=None):
        self.num_times=len(slices)
        for edges_t in slices:
            assert edges_t.shape[0]>0
            assert edges_t.shape[1]==2

        if num_nodes is None:
            num_nodes = max(map(np.max, slices)) + 1
        self.slices=[FastGraph(v, is_directed=is_directed, num_nodes=num_nodes) for v in slices]
        self.num_nodes = max(G.num_nodes for G in self.slices)
        if times is None:
            times = np.arange(self.num_times)
        self.times = times
        self.is_directed = is_directed
        if r is None:
            r = len(self.slices)
        self.r = r


    def to_temporal_edges(self, base_edges=False):
        """Returns the graph represented as temporal edges
        a temporal edge is a triple (u->v, t)
        """
        total_number_of_edges = sum(len(G.edges) for G in self.slices)
        E = np.empty((total_number_of_edges, 3), dtype=np.int64)
        n = 0
        for t, G in zip(self.times, self.slices):
            if base_edges:
                partial_edges = G.base_edges
            else:
                partial_edges = G.edges
            E[n:n+len(partial_edges), 0:2] = partial_edges
            E[n:n+len(partial_edges), 2] = t
            n+=len(partial_edges)
        return E

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

    def apply_wl_colors_to_slices(self, wl_colors=None):
        """Assign wl colors to each slice"""
        if wl_colors is None:
            wl_colors = self.calc_wl()
        wl_colors_by_slice = [[] for _ in range(len(self.slices))]
        for colors_per_depth in wl_colors:
            for i, slice_colors in enumerate(wl_colors_by_slice):
                slice_colors.append(colors_per_depth[i,:].ravel())
        for G, colors in zip(self.slices, wl_colors_by_slice):
            G.base_partitions = np.array(colors, dtype=np.int32)
            G.wl_iterations = len(G.base_partitions)
            G.reset_edges_ordered()

    def copy(self):
        """Creates a copy of this temporal graph"""
        new_slices = [s.edges.copy() for s in self.slices]
        return TempFastGraph(new_slices, self.is_directed, self.times.copy(), num_nodes=self.num_nodes)


    def rewire(self, depth, method, **kwargs):
        """Rewires each slice of the temporal graph"""
        if self.is_directed:
            raise NotImplementedError("Not yet implemented for directed graphs")
        for s in self.slices:
            s.rewire(depth, method=method, **kwargs)

    def get_all_partitions(self):
        """Returns all partitions of the entire graph stacked"""
        if self.slices[0].base_partitions is None:
            cols = self.calc_wl()
            self.apply_wl_colors_to_slices(cols)
        partitions = np.hstack([G.base_partitions for G in self.slices])
        indentifiers = [(v, t) for v,t in product(range(self.num_nodes), range(len(self.times)))]
        return indentifiers, partitions




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
        first_index = np.zeros(self.num_nodes, dtype=np.int32)
        last_index = np.zeros(self.num_nodes, dtype=np.int32)
        for t in range(T-1, -1, -1):
            g = self.slices[t]

            num_total_edges = len(g.edges) + np.sum(last_index - first_index)

            for u, v in g.edges:
                buff_size = len(list_successors[u])
                u_successors = list_successors[u]
                u_successors[last_index[u] % buff_size] = v + t*self.num_nodes
                time_successors[u][last_index[u] % buff_size] = t
                last_index[u] +=1
            new_edges = np.empty((num_total_edges, 2), dtype=np.int32)
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
        adjacencies = [G.global_to_coo() for G in self.slices]
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
    mapping = {key : val for key, val in zip(unique, range(len(unique)))}
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
    return {gloabl_node_id : values[local_node_id] for gloabl_node_id, local_node_id in mapping.items()}



class MappedGraph(FastGraph):
    """A FastGraph that removes degree zero nodes through relabeling."""
    def __init__(self, raw_edges, is_directed, mapped_edges, mapping, unmapping, global_num_nodes, internal_num_nodes=None):
        self.raw_edges = raw_edges
        self.mapping = mapping
        self.unmapping = unmapping
        self.global_nodes = np.fromiter(mapping.keys(), count=len(self.mapping), dtype=np.int64)
        self.global_num_nodes = global_num_nodes

        super().__init__(mapped_edges, is_directed, num_nodes=internal_num_nodes)

    @classmethod
    def from_edges(cls, edges, is_directed, global_num_nodes=None):
        """Create a temporal graph from themporal edges"""
        raw_edges = edges.copy()
        mapped_edges, mapping, unmapping = relabel_edges(edges)
        internal_num_nodes = len(unmapping)
        return cls(raw_edges, is_directed, mapped_edges, mapping, unmapping, global_num_nodes=global_num_nodes, internal_num_nodes=internal_num_nodes)


    @property
    def local_edges(self):
        """Returns the edges of the underlying FastGraph object"""
        return super().edges

    @property
    def global_edges(self):
        """Returns the edges in the global name space"""
        return apply_mapping_to_edges(super().edges, self.unmapping)

    @property
    def base_global_edges(self):
        """Returns the edges in the global name space"""
        return apply_mapping_to_edges(self._edges, self.unmapping)

    @property
    def sparse_out_degree(self):
        """Returns the out degree as a dictionary"""
        return to_mapping(self.out_degree, self.mapping)

    @property
    def sparse_in_degree(self):
        """Returns the in degree as a dictionary"""
        return to_mapping(self.in_degree, self.mapping)


    def global_to_coo(self):
        """Returns a sparse coo-matrix representation of the graph"""
        from scipy.sparse import coo_matrix # pylint: disable=import-outside-toplevel
        edges = self.global_edges
        if not self.is_directed:
            edges = make_directed(edges)

        return coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape = (self.global_num_nodes, self.global_num_nodes))

    def switch_directions(self):
        """Creates a FastGraph object from a graphtool graph"""

        return MappedGraph(switch_in_out(self.raw_edges),
                           self.is_directed,
                           switch_in_out(self.edges),
                           mapping=self.mapping,
                           unmapping=self.unmapping,
                           global_num_nodes=self.global_num_nodes,
                           internal_num_nodes=self.num_nodes)



def _get_rolling_max_degree(list_degrees, list_mapping, r, num_nodes):

    # r = 1 is that we we look at the current and the next slice
    # thus need to increase r by 1 to get buf size
    buff_size = r+1
    T = len(list_degrees)
    max_degree = np.zeros(num_nodes, dtype=np.int32)
    curr_degree = np.zeros(num_nodes, dtype=np.int32)
    roll_degree = np.zeros((num_nodes, buff_size), dtype=np.int32)
    roll_time = np.zeros((num_nodes, buff_size), dtype=np.int32)
    first_index = np.zeros(num_nodes, dtype=np.int32)
    last_index = np.zeros(num_nodes, dtype=np.int32)
    for t in range(T):
        mapping = list_mapping[t]
        degrees = list_degrees[t]
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


def get_rolling_max_degree(l_degrees, l_mapping, is_directed, h, num_nodes):
    """Computes the maximum degree of a node (i.e. summed over time) for a finite horizon h


    """
    assert len(l_degrees[0])==len(l_mapping)
    assert h >= 0
    if is_directed:
        max_in_degree = _get_rolling_max_degree(l_degrees[0], l_mapping, h, num_nodes)
        max_out_degree = _get_rolling_max_degree(l_degrees[1], l_mapping, h, num_nodes)
        return max_in_degree, max_out_degree
    else:
        max_degree = _get_rolling_max_degree(l_degrees[0], l_mapping, h, num_nodes)
        return max_degree, max_degree


def get_total_degree(l_edges, is_directed, num_nodes):
    """Computes the total degree of a node (i.e. summed over time)"""
    if is_directed:
        in_degree = np.zeros(num_nodes, dtype=np.int32)
        out_degree = np.zeros(num_nodes, dtype=np.int32)
        for edges in l_edges:
            for i in range(len(edges)):
                out_degree[edges[i,0]]+=1
                in_degree[edges[i,1]]+=1
        return in_degree, out_degree
    else:
        degree = np.zeros(num_nodes, dtype=np.int32)
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
    def __init__(self, edges, is_directed, num_nodes=None, times=None):
        for edges_t in edges:
            assert edges_t.shape[0]>0
            assert edges_t.shape[1]==2
        self.num_times=len(edges)
        if num_nodes is None:
            num_nodes = max(map(np.max, edges)) + 1
        self.num_nodes = num_nodes
        self.slices=[MappedGraph.from_edges(edges_t, is_directed=is_directed, global_num_nodes=num_nodes) for edges_t in edges]
        self.is_directed = is_directed
        if times is None:
            self.times=np.arange(len(self.slices))
        else:
            self.times=np.asanyarray(times).ravel()
        assert self.num_times == len(self.times)
        self.h = None


    @staticmethod
    def from_temporal_edges(E, is_directed, num_nodes=None):
        """Creates a sparse temporal graph from temporal edges"""
        assert len(E.shape)==2
        assert E.shape[1]==3
        edges, times = partition_temporal_edges(E)
        return SparseTempFastGraph(edges, is_directed=is_directed, num_nodes=num_nodes, times=times)


    def to_temporal_edges(self, base_edges=False):
        """Returns the graph represented as temporal edges
        a temporal edge is a triple (u->v, t)
        """
        total_number_of_edges = sum(len(G.edges) for G in self.slices)
        E = np.empty((total_number_of_edges, 3), dtype=np.int64)
        n = 0
        for t, G in zip(self.times, self.slices):
            if base_edges:
                partial_edges = G.base_global_edges
            else:
                partial_edges = G.global_edges
            E[n:n+len(partial_edges), 0:2] = partial_edges
            E[n:n+len(partial_edges), 2] = t
            n+=len(partial_edges)
        return E

    def to_dense(self):
        """Returns a dense temporal graph fro this sparse temporal graph"""
        return TempFastGraph([_slice.global_edges for _slice in self.slices], self.is_directed, self.times, num_nodes=self.num_nodes)


    def reverse_slice_direction(self):
        """Returns a new graph with all edge directions reversed within time slices"""
        reversed_edges = [G.switch_directions().global_edges for G in self.slices]
        return SparseTempFastGraph(reversed_edges, is_directed=self.is_directed, num_nodes=self.num_nodes)


    def reverse_time(self):
        """Returns a new graph with time reversed"""
        reversed_edges = [G.global_edges for G in reversed(self.slices)]
        return SparseTempFastGraph(reversed_edges, is_directed=self.is_directed, num_nodes=self.num_nodes)


    def get_temporal_wl_struct(self, h=-1, d=-1, seed=0, kind="broadcast", base_edges=False):
        """Computes the temporal WL and assigns it to each graph"""
        edges = self.to_temporal_edges(base_edges=base_edges)
        if not self.is_directed:
            edges2 = np.vstack((edges[:,1], edges[:,0], edges[:,2])).T
            edges = np.vstack((edges, edges2))
        else:
            pass
            #rev_edges = np.vstack((edges[:,1], edges[:,0], edges[:,2])).T

        if kind=="broadcast":
            return TemporalColorsStruct(*_compute_d_rounds(edges, self.num_nodes, d=d, h=h, seed=seed))
        elif kind=="receive":
            raise NotImplementedError()


    def assign_colors_to_slices(self, h=-1, d=-1, seed=0, sorting_strategy="auto", kind="broadcast", mode="global"):
        """Assign the temporal wl colors to individual slices"""
        s = self.get_temporal_wl_struct(h, d, seed, kind)

        max_d = len(s.colors_per_round)
        print("max_d", max_d)
        for d_iter in range(max_d):
            s.reset_colors(d_iter, mode=mode)
            for t, G in zip(self.times, self.slices):
                if d_iter == 0:
                    G.base_partitions = []
                s.advance_time(t)
                G.base_partitions.append(s.current_colors[G.global_nodes])
        if sorting_strategy == "auto":
            if self.is_directed:
                sorting_strategy = "source"
            else:
                sorting_strategy = "both"
        for G in self.slices:
            G.base_partitions = np.array(G.base_partitions, dtype=np.int32)
            G.reset_edges_ordered(sorting_strategy)
        self.h = h
        self.s=s
        return s


    def rewire_slices(self, depth, method, **kwargs):
        """Rewires all temporal slices using previously assigned colors"""
        assert not self.h is None, "You need to provide wl colors"
        for G in self.slices:
            G.rewire(depth=depth, method=method, **kwargs)


    def compute_for_each_slice(self, func, min_size=None, fill_value=None, call_with_time=True, dtype=np.float64, auto_resize=True):
        """Helper functions that allows to compute a function for each slice

        If call_with_time=True two arguments are provided (the graph G, and time t)
        If call_with_time=False one argument is provided (the graph G)

        the function automatically unpacks provided arguments

        """
        if fill_value is None:
            # use nans for floats and zeros for ints
            if dtype == np.float32 or dtype==np.float64:
                fill_value = np.nan
            else:
                fill_value = 0
        if min_size is None:
            min_size = 1
        arr = np.full((len(self.slices), min_size), fill_value=fill_value, dtype=dtype)
        has_warned = False
        for i, (G, t) in enumerate(zip(self.slices, self.times)):
            if call_with_time:
                func_values = func(G, t)
            else:
                func_values = func(G)
            if isinstance(func_values, (list, tuple, np.ndarray)):
                if len(func_values) <= arr.shape[1]:
                    arr[i, :len(func_values)] = func_values
                else:
                    if auto_resize:
                        tmp_arr = np.full((len(self.slices), len(func_values)),fill_value=fill_value, dtype=arr.dtype)
                        tmp_arr[:i,arr.shape[1]] = arr
                        tmp_arr[i,:] = func_values
                        arr = tmp_arr
                    else:
                        raise ValueError(f"Inconsistent shape found for timeslice {i} at time {t}")
            elif np.isscalar(func_values):
                arr[i,0] = func_values
                if arr.shape[1]!=1 and not has_warned:
                    has_warned = True
                    warnings.warn(f"Inconsistent shape found for timeslice {i} at time {t}", RuntimeWarning)
            else:
                raise ValueError(f"func returned the type {type(func_values)} which are currently not supported")
        if arr.shape[1]==1:
            arr = arr.ravel()
        return self.times, arr


    def get_dense_causal_completion(self, h=-1):
        """Return the dense causal completion"""
        E_temp = self.to_temporal_edges()
        if not self.is_directed:
            E_temp = temp_undir_to_directed(E_temp)
        E_out = get_edges_dense_causal_completion(E_temp, self.times, self.num_nodes, h=h)
        G = FastGraph(E_out, is_directed=True, num_nodes=len(self.times)*self.num_nodes)
        G.identifiers = get_dense_identifiers(self.times, self.num_nodes)
        return G






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
            local_edges = g.local_edges
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

            edges = np.empty((total_edges,2), dtype=np.int32)
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
                    node_identifier[global_name,0] = g.unmapping[i]
                    node_identifier[global_name,1] = t
                    n_temp+=1

        all_edges = np.vstack(list(reversed(all_edges)))
        G = FastGraph(all_edges, is_directed=True, num_nodes=num_prev_nodes[-1])
        G.identifiers = node_identifier
        if return_temporal_info:
            return G, (self.num_nodes, T)
        else:
            return G
