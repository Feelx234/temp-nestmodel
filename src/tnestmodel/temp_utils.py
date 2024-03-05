import numpy as np
from numba import njit

@njit(cache=True)
def temp_undir_to_directed(E):
    """Converts undirected temporal edges to directed temporal edges"""
    E_out = np.empty((2 * len(E), 3), dtype=E.dtype)
    n = 0
    for u,v, t in E:
        E_out[n,0]=u
        E_out[n,1]=v
        E_out[n,2]=t

        E_out[n+1,0]=v
        E_out[n+1,1]=u
        E_out[n+1,2]=t
        n+=2
    return E_out



@njit(cache=True)
def partition_temporal_edges(E):
    """Partitions temporal edges into graphs per time"""
    time_order = np.argsort(E[:,2])
    E = E[time_order, :]
    
    
    last_t = E[0,2]
    curr_list = [(E[0,0], E[0,1])]
    curr_list.pop()
    empty_list = curr_list.copy()
    list_edges = [curr_list]
    times = [last_t]
    for u,v,t in E:
        if t != last_t:
            curr_list = empty_list.copy()
            list_edges.append(curr_list)
            last_t = t
            times.append(last_t)
        curr_list.append((u,v))
 
    list_of_arrs = [np.array(x, dtype=np.uint32) for x in list_edges]
    return list_of_arrs, times