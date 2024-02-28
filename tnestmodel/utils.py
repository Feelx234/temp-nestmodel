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
