import numpy as np
from nestmodel.mutual_independent_models import Gnp_row_first
from tnestmodel.temp_fast_graph import SparseTempFastGraph

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

