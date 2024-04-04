from collections import defaultdict
import numpy as np
from numba import njit
from nestmodel.fast_graph import FastGraph
from tnestmodel.wl_utils import assert_partitions_equivalent
from tnestmodel.temp_utils import temp_undir_to_directed

@njit(cache=True)
def calculate_number_of_dense_edges(E_temp):
    """Calculate the number of edges that will be present in the dense causal completion"""
    s = 0
    last_t = E_temp[0,2]
    t_index = 1
    for (_, _, t) in E_temp:
        if t != last_t:
            last_t = t
            t_index+=1
        s+=t_index
    return s

@njit(cache=True)
def get_dense_identifiers(times, num_nodes):
    """Returns a 2d array of identifiers
    shape of output is (num_notes*len(times), 2)
    the first entry in each row is the time
    the second entry is the non temporal node identifier
    """
    num_all_nodes = len(times) * num_nodes
    identifiers = np.empty((num_all_nodes, 2), dtype=np.int64)
    n = 0
    for t in range(len(times)):
        for v in range(num_nodes):
            identifiers[n,0] = v
            identifiers[n,1] = t
            n+=1
    return identifiers


@njit(cache=True)
def get_edges_dense_causal_completion(E_temp, times, num_nodes, h):
    """Returns the edge_set for the dense causal completion

    Assumes that E_temp is increasing in time
    """
    total_num_edges = calculate_number_of_dense_edges(E_temp)
    E_out = np.empty((total_num_edges, 2), dtype=np.uint32)
    n=0
    last_t = times[0]
    t_index = 0
    start_index = 0
    for j in range(E_temp.shape[0]):
        u = E_temp[j,0]
        v = E_temp[j,1]
        t = E_temp[j,2]
        if t != last_t:
            last_t = t
            t_index+=1
        curr_u = u+t_index*num_nodes
        curr_v = v+t_index*num_nodes

        E_out[n,0]=curr_u
        E_out[n,1]=curr_v
        n+=1
        for i in range(start_index, t_index):
            if h>=0 and times[i] < t-h:
                # print("skipped", t, h, times[i])
                start_index = i+1
                continue
            E_out[n,0]=u+i*num_nodes
            E_out[n,1]=curr_v
            n+=1
    return E_out[:n,:]




def get_potentially_active_nodes(E, h, num_nodes):
    """Returns an array of potentially active nodes of the temporal graph specified by the temporal edges E"""
    pactive_nodes = []
    times = np.unique(E[:,2])

    next_after={}
    for i,t in enumerate(times[:-1]):
        next_after[t]=times[i+1]
    last_time = times[-1]
    # process edges no longer visible
    #  we find all nodes that loose an edge by increasing the time
    #  for these we increase the left active node counter which
    #  will subsequently affect that particular nodes hash
    right_e_index=0
    last_active_time = np.full(num_nodes, -1, dtype=np.int64)
    for (u, v, t) in E:
        if last_active_time[u]<t:
            pactive_nodes.append((u,t))
            last_active_time[u]=t
        if last_active_time[v]<t:
            pactive_nodes.append((v,t))
            last_active_time[v]=t
        if t<last_time:
            pactive_nodes.append((u, next_after[t]))

    t0=times[0]
    last_active_time[:]=-1
    for t in times:
        # process newly visible edges
        #  by advancing the time, we add newly visible edges
        while right_e_index < len(E):
            (u, v, t_e) = E[right_e_index,:]
            if t_e > t + h:
                # we no longer see this edge or any future edges, break
                break
            if last_active_time[u]<t:
                # print("adding", u, t)
                pactive_nodes.append((u,t))
                last_active_time[u]=t
            right_e_index+=1
    pactive_nodes = np.array(pactive_nodes, dtype=np.int64)
    order=np.lexsort(pactive_nodes.T)
    return pactive_nodes[order,:]

def remove_consecutive_duplicates(pactive_nodes):
    """Removes consecutive duplicated from pactive_nodes inplace!"""
    n = 1
    last_u, last_t = pactive_nodes[0,:]
    for i,(u,t) in enumerate(pactive_nodes[1:,:]):
        if last_u==u and last_t==t:
            continue
        last_u=u
        last_t=t
        pactive_nodes[n,0] = u
        pactive_nodes[n,1] = t
        n+=1
    return pactive_nodes[:n,:]



def collect_out_edges_per_node(E):
    """Returns a dictionary that contains for each node v a list of temporal nodes that are out neighbors of v"""
    per_node = defaultdict(list)
    for u,v,t in E:
        per_node[u].append((v,t))
        len(per_node[v])
    for key in per_node.keys():
        per_node[key]=np.array(per_node[key])
    return per_node

def _create_sparse_causal_graph(per_node, pactive_nodes, h, num_nodes):
    E_out = []
    tuple_to_int = {(v,t): i for i, (v,t) in enumerate(pactive_nodes)}
    left_per_node = [0]*num_nodes
    right_per_node = [0]*num_nodes
    neighbors_int = {u: np.array([tuple_to_int[(v,t)] for v,t in per_node[u]]) for u in per_node.keys()}
    for v,t in pactive_nodes:
        v_out = tuple_to_int[(v,t)]
        potential_neighbors = per_node[v]
        while left_per_node[v]< len(potential_neighbors) and potential_neighbors[left_per_node[v]][1]<t:
            left_per_node[v]+=1
        while right_per_node[v]< len(potential_neighbors) and potential_neighbors[right_per_node[v]][1]<=t+h:
            right_per_node[v]+=1
        to_append = np.empty((right_per_node[v]-left_per_node[v],2), dtype=np.uint32)
        to_append[:,0]=v_out
        to_append[:,1]=neighbors_int[v][left_per_node[v]:right_per_node[v]]
        E_out.append(to_append)
        #E_out.append(nerighbors_int[v][left_per_node[v]:right_per_node[v]].copy())
        #for i in range(left_per_node[v], right_per_node[v]):
        #    E_out.append((v_out, tuple_to_int[potential_neighbors[i]]))
    int_to_tuple = {i : (v,t) for  (v,t), i in tuple_to_int.items()}
    return np.vstack(E_out), int_to_tuple




def create_sparse_causal_graph(E_temp, h, is_directed, num_nodes, should_print=False):
    """Create a sparse causal graph from temporal edges and a given horizon h"""
    if not is_directed:
        E_temp = temp_undir_to_directed(E_temp)
    pactive_nodes = get_potentially_active_nodes(E_temp, h, num_nodes)
    if should_print:
        print(len(pactive_nodes))
    pactive_nodes = remove_consecutive_duplicates(pactive_nodes)
    if should_print:
        print(len(pactive_nodes))

    num_all_nodes = len(pactive_nodes)
    per_node = collect_out_edges_per_node(E_temp)
    E_out, int_to_tuple = _create_sparse_causal_graph(per_node, pactive_nodes, h, num_nodes)
    return E_out, num_all_nodes, int_to_tuple


def sparse_wl_to_dense_wl(G_sparse, colors, add_nodes=0):
    """Convert the wl colors computed for the sparse causal graph to dense wl colors
    if add_nodes is provided, it is assumed, that the last add_nodes nodes are degree zero nodes that can be ignored.

    """
    if add_nodes >0:
        zero_color = colors[-add_nodes]
        colors=colors[:-1]
    else:
        colors_of_zero_nodes = colors[G_sparse.in_degree==0]
        if len(colors_of_zero_nodes)>0:
            zero_color = colors_of_zero_nodes[0]
            assert np.all(zero_color==colors_of_zero_nodes)
        else:
            zero_color = np.max(colors)+1
    nodes = np.unique(G_sparse.identifiers[:,0].ravel())
    times = np.unique(G_sparse.identifiers[:,1].ravel())
    time_to_index = {t:i for i,t in enumerate(times)}
    num_nodes = nodes[-1]+1
    num_times = len(times)
    dense_colors = np.full((num_times, num_nodes), zero_color, dtype=np.int32)
    times_colors_per_node = defaultdict(list)
    for (v,t), color in zip(G_sparse.identifiers, colors):
        times_colors_per_node[v].append((t, color))
    for v, l in times_colors_per_node.items():
        for i in range(len(l)-1):
            t = time_to_index[l[i][0]]
            next_t = time_to_index[l[i+1][0]]
            color = l[i][1]
            assert t < next_t
            # print(v, "", t, next_t, color)

            dense_colors[t:next_t,v] = color
        t = time_to_index[l[len(l)-1][0]]
        color = l[len(l)-1][1]
        # print(v, "", t, color)
        dense_colors[t,v] = color
    return dense_colors


def identifiers_from_int_to_tuple(int_to_tuple):
    """Create the identifiers array from in_to_tuple mapping"""
    identifiers = np.zeros((len(int_to_tuple),2), dtype=np.int32)
    for key, (v,t) in int_to_tuple.items():
        identifiers[key,0] = v
        identifiers[key,1] = t
    return identifiers


def get_sparse_causal(G_temp, h, add_nodes=0):
    """Obtain a sparse causal graph from the temporal graph G_temp at horizon h"""
    E = G_temp.to_temporal_edges()
    E_out, num_all_nodes, int_to_tuple = create_sparse_causal_graph(E, h, False, G_temp.num_nodes)
    G = FastGraph(np.array(E_out, dtype=np.uint32), is_directed=True, num_nodes=num_all_nodes+add_nodes).switch_directions()
    G.identifiers = identifiers_from_int_to_tuple(int_to_tuple)
    G.num_nodes_per_time=G_temp.num_nodes
    return G


def compare_wl_dense_sparse(G_temp, h):
    """Asserts that the dense and sparse causal graphs produce the same WL colors for graph G_temp"""
    dense_colors = get_all_dense_wl_colors(G_temp, h)

    # print(tmp_sparse_colors, sparse_colors, dense_colors)
    sparse_colors = get_all_sparse_wl_colors(G_temp, h)
    # [x.reshape((len(G_temp.times), G_temp.num_nodes)) for x in sparse_colors]
    assert len(sparse_colors) == len(dense_colors), f"Number of iterations does not match h = {h}"

    for dense, sparse in zip(dense_colors, sparse_colors):
        assert_partitions_equivalent(dense, sparse)


def compare_wl_dense_cumsum(G_temp, h):
    """Asserts that the dense and sparse causal graphs produce the same WL colors for graph G_temp"""
    dense_colors = get_all_dense_wl_colors(G_temp, h)
    cumsum_colors = get_all_cumsum_wl_colors(G_temp, h)
    # [x.reshape((len(G_temp.times), G_temp.num_nodes)) for x in sparse_colors]
    #assert len(cumsum_colors) == len(dense_colors), f"Number of iterations does not match h = {h}"

    for dense, cumsum in zip(dense_colors, cumsum_colors):
        assert_partitions_equivalent(dense, cumsum)

def get_all_sparse_wl_colors(G_temp, h):
    add_nodes=1
    G_sparse = get_sparse_causal(G_temp, h=h, add_nodes=add_nodes)
    tmp_sparse_colors = G_sparse.calc_wl()
    sparse_colors = [sparse_wl_to_dense_wl(G_sparse, sparse, add_nodes=add_nodes) for sparse in tmp_sparse_colors]
    return [x.ravel() for x in sparse_colors]

def get_all_dense_wl_colors(G_temp, h):
    G_tmp = G_temp.get_dense_causal_completion(h=h)
    G_dense  = G_tmp.switch_directions()
    G_dense.identifiers = G_tmp.identifiers
    return G_dense.calc_wl()

def get_all_cumsum_wl_colors(G_temp, h):
    s = G_temp.get_temporal_wl_struct(h=h)
    out = []
    for d in range(len(s.cumsum_hashes_per_round)+1):
        colors= s.get_colors_all_times(d=d, h=h)
        out.append(colors)
    return out