import numpy as np
from numba import njit
from nestmodel.fast_rewire import _set_seed
from nestmodel.mutual_independent_models import Gnp_row_first
from tnestmodel.temp_fast_graph import SparseTempFastGraph
from tnestmodel.temp_properties import _get_aggregated_graph, _get_aggregated_graph_from_edges


def temporal_Gnp(n, p, times, seed=0):
    """Create a temporal graph which has an erdos reny graph at each slice"""
    if isinstance(times, int):
        times = np.arange(times)
    elif isinstance(times, (tuple, list)):
        times = np.array(times, dtype=np.int32)
    else:
        raise ValueError(f"times has type {type(times)} with value {times} which is not supported")

    num_times = len(times)
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
        seeds = rng.integers(low=0, high=100*num_times, size=num_times)
    elif seed is None:
        seeds = np.random.randint(low=0, high=100*num_times, size=num_times)
    else:
        raise ValueError(f"seed has type {type(seed)} with value {seed} which is not supported")

    temporal_edges = []
    for seed, t in zip(seeds, times, strict=True):
        edges = Gnp_row_first(n, p, seed)
        t_edges = np.empty((len(edges),3), dtype=edges.dtype)
        t_edges[:,:2]=edges
        t_edges[:,2] = t
        temporal_edges.append(t_edges)

    temporal_edges = np.vstack(temporal_edges)
    return SparseTempFastGraph.from_temporal_edges(temporal_edges, is_directed=False, num_nodes=n)




@njit
def sample_without_replacement2(arr, k):
    """Sample k values without replacement from arr avoiding to sample avoid

    Avoid is assumed to appear no more than once in arr.

    This mutates arr!
    The algorithm used is a variant of the Fisher-Yates shuffle
    """
    n = len(arr)

    if k==len(arr):
        return arr.copy()
    if 2*k <= n:
        num_select = k # choose k elements and put them to the front
    else:
        num_select = n-k # choose n-k elements and put them to the front
                         # these elements will be excluded
    for j in range(num_select):
        val = j + np.random.randint(0,n-j)
        tmp = arr[j]
        arr[j] = arr[val]
        arr[val] = tmp
    if k<= n//2:
        return arr[:k].copy()
    else:
        return arr[k:].copy() # return the included elements
    


def randomize_keeping_aggregate_graph(d, times, keep_multiplicities=True, seed=None):
    if not seed is None:
        _set_seed(seed) # numba seed
        np.random.seed(seed) # numpy seed, seperate from numba seed
    return _randomize_keeping_aggregate_graph(d, times, keep_multiplicities)



@njit
def _randomize_keeping_aggregate_graph(d, times, keep_multiplicities):
    d = d.copy()
    M = np.empty((len(d),3), dtype=np.int64)
    i = 0
    for (u,v), m in d.items():
        M[i,0] = u
        M[i,1] = v
        M[i,2] = m
        i+=1
    
    num_temporal_edges = M[:,2].sum()  
    if not keep_multiplicities:
        M[:,2]=1
        remaining_times = num_temporal_edges - M.shape[0]
        for _ in range(remaining_times):
            j = np.random.randint(0, M.shape[0])
            M[j,2]+=1
    
    i = 0
    E = np.empty((num_temporal_edges, 3), dtype=np.int64)
    rand_times = times.copy()
    for k in range(M.shape[0]):
        u=M[k,0]
        v=M[k,1]
        m=M[k,2]
        sample = sample_without_replacement2(rand_times, m)
        for j in range(m):
            E[i,0] = u
            E[i,1] = v
            E[i,2] = sample[j]
            i+=1
    return E

def assert_agg_identical(E, d1, is_directed, keep_multiplicities):
    """Asserts that the generated edges E and the aggregated graph d1 are the valid"""
    d2 = _get_aggregated_graph_from_edges(E, is_directed)
    assert len(d1)==len(d2), f"{len(d1)}, {len(d2)}"
    if keep_multiplicities:
        for (u,v), m in d1.items():
            assert d2[(u,v)]==m, f"{(u,v)}, mults= {d2[(u,v)]} {m} {d2[(v,u)]}"
    else:
        s1 = sum(d1.values())
        s2 = sum(d2.values())
        assert s1 == s2
        for (u,v), m in d1.items():
            assert (u,v) in d2
        