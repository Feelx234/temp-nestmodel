import numpy as np
from numba import njit
from numba.typed import Dict  #pylint: disable=no-name-in-module
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
    if 2*k <= n:
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


def undir_stub_matching(G, seed=None, max_tries=1000):
    assert not G.is_directed
    if seed is not None:
        np.random.seed(seed)
    E = G.to_temporal_edges()
    num_edges = E.shape[0]
    tmp = np.empty(num_edges*2, dtype=E.dtype)
    tmp[0:num_edges] = E[:,0]
    tmp[num_edges:] = E[:,1]
    for num_tries in range(max_tries):
        tmp = tmp[np.random.permutation(2*num_edges)]
        out = tmp.reshape(num_edges, 2)
        if np.all(out[:,0]!=out[:,1]):
            break
        out = None
    return out


@njit
def _dir_very_random_temp_rewiring(E, n_steps):
    """Rewires a directed temporal graph using edge switches"""
    successes=0
    d = Dict()
    for u,v,t in E:
        d[(u,v,t)]=1
    num_edges = E.shape[0]
    for _ in range(n_steps):
        i1 = np.random.randint(0, num_edges)
        i2 = np.random.randint(0, num_edges)
        if i1==i2:
            continue

        u1 = E[i1, 0]
        v1 = E[i1, 1]
        t1 = E[i1, 2]

        u2 = E[i2, 0]
        v2 = E[i2, 1]
        t2 = E[i2, 2]

        flip_time = np.random.randint(0, 2)

        if flip_time==1:
            new_t1 = t2
            new_t2 = t1
        else:
            new_t1 = t1
            new_t2 = t2
        if (u1, v2, new_t1) not in d and (u2, v1, new_t2) not in d:
            del d[(u1, v1, t1)]
            del d[(u2, v2, t2)]
            d[(u1, v2, new_t1)]=1
            d[(u2, v1, new_t2)]=1

            E[i1, 1] = v2
            E[i1, 2] = new_t1
            E[i2, 1] = v1
            E[i2, 2] = new_t2

            successes +=1
    return successes


def very_random_temp_rewiring(G, n_steps, seed=None, E = None):
    if not seed is None:
        _set_seed(seed) # numba seed
        np.random.seed(seed) # numpy seed, seperate from numba seed
    if E is None:
        E = G.to_temporal_edges()
    if G.is_directed:
        attempt_steps = max(int(n_steps//len(E)), 1)
        successes = dir_very_random_tilt(E, attempt_steps, G.num_nodes, G.times)
    else:
        successes = _undir_very_random_temp_rewiring(E, n_steps)
    return E, successes




@njit
def _undir_very_random_temp_rewiring(E, n_steps):
    successes=0
    d = Dict()
    for u,v,t in E:
        d[(u,v,t)]=1
    num_edges = E.shape[0]
    for _ in range(n_steps):
        i1 = np.random.randint(0, num_edges)
        i2 = np.random.randint(0, num_edges)
        if i1==i2:
            continue

        u1 = E[i1, 0]
        v1 = E[i1, 1]
        t1 = E[i1, 2]

        u2 = E[i2, 0]
        v2 = E[i2, 1]
        t2 = E[i2, 2]

        flip_time = np.random.randint(0, 2)

        if flip_time==1:
            new_t1 = t2
            new_t2 = t1
        else:
            new_t1 = t1
            new_t2 = t2

        flip_edge = np.random.randint(0, 2)

        if flip_edge==1:
            tmp = u1
            u1 = v1
            v1 = tmp
        if (u1, v2, new_t1) not in d and (u2, v1, new_t2) not in d and (v2, u1, new_t1) and (v1, u2, new_t2):
            if flip_edge==1:
                del d[(v1, u1, t1)]
            else:
                del d[(u1, v1, t1)]
            del d[(u2, v2, t2)]
            d[(u1, v2, new_t1)]=1
            d[(u2, v1, new_t2)]=1

            E[i1, 0] = u1
            E[i1, 1] = v2
            E[i1, 2] = new_t1

            E[i2, 0] = u2
            E[i2, 1] = v1
            E[i2, 2] = new_t2
            successes +=1
    return successes

@njit
def dir_very_random_tilt(E, num_attempts, num_nodes, times):
    successes=0
    d = Dict()
    for u,v,t in E:
        d[(u,v,t)]=1
    num_edges = E.shape[0]
    num_times = len(times)
    for i in range(num_edges):
        u1 = E[i, 0]
        v1 = E[i, 1]
        t1 = E[i, 2]
        for _ in range(num_attempts):
            i_t = np.random.randint(0, num_times)
            new_t = times[i_t]
            v2 = np.random.randint(0, num_nodes)
            if (u1, v2, new_t) not in d:
                # print((u1, v1, t1), (u1, v2, new_t))
                del d[(u1, v1, t1)]
                d[(u1, v2, new_t)]=1
                E[i,0]=u1
                E[i,1]=v2
                E[i,2]=new_t

                successes+=1
                break
    return successes