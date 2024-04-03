import numpy as np
from numba import njit

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
    for t in times:
        for v in range(num_nodes):
            identifiers[n,0] = t
            identifiers[n,1] = v
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