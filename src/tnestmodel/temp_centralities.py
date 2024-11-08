# pylint: disable=import-outside-toplevel, missing-function-docstring
import numpy as np
from nestmodel.unified_functions import get_sparse_adjacency, num_nodes
from nestmodel.centralities import calc_katz, calc_katz_iter
from tnestmodel.temp_fast_graph import TempFastGraph, SparseTempFastGraph
from tnestmodel.temp_fast_graph import MappedGraph
def is_t_fastgraph_str(G_str):
    return (G_str.startswith("<tnestmodel.temp_fast_graph.SparseTempFastGraph") or
            G_str.startswith("<tnestmodel.temp_fast_graph.TempFastGraph "))

def t_num_nodes(G):
    G_str = repr(G)
    if is_t_fastgraph_str(G_str):
        return G.num_nodes
    return num_nodes(G)

def t_sparse_adjacency(G):
    if isinstance(G, MappedGraph):
        return G.global_to_coo()
    return get_sparse_adjacency(G)


KIND_BROADCAST = "broadcast"
KIND_RECEIVE = "receive"

def calc_temp_katz(G_t, alpha=0.1, epsilon=0, max_iter=None, kind=KIND_BROADCAST): # pylint: disable=unused-argument
    """Returns the katz scores of the current graph"""
    from scipy.sparse.linalg import spsolve # pylint: disable=import-outside-toplevel
    from scipy.sparse import identity

    assert kind in (KIND_BROADCAST, KIND_RECEIVE)

    n = t_num_nodes(G_t)
    b=np.ones(n)
    if hasattr(G_t, "r"):
        assert G_t.r >= len(G_t.slices), "Katz centrality is not supported for restless walks, use causal method instead."

    if kind==KIND_BROADCAST:
        slices_in_right_order = reversed(G_t.slices)
    else:
        slices_in_right_order = G_t.slices
    for G in slices_in_right_order:
        A = t_sparse_adjacency(G)
        A.resize(n,n)

        if kind==KIND_RECEIVE:
            A = A.T

        A = identity(n) - alpha * A

        katz = spsolve(A, b, )
        b = katz

    return katz


def get_leftmost_entry_for_nodes(katz, identifiers, n, default=1):
    out = np.full(n, default, dtype=np.float64)
    min_time = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    for i in range(len(katz)): # pylint: disable=consider-using-enumerate
        t = identifiers[i, 1]
        node = identifiers[i, 0]
        if t < min_time[node] and katz[i]>1.0:
            out[node]=katz[i]
            min_time[node]=t
    return out


def calc_temp_katz_from_causal(G_t, alpha=0.1, epsilon=0, max_iter=None, kind="broadcast"): # pylint: disable=unused-argument
    T = len(G_t.slices)
    n = G_t.num_nodes

    if kind == KIND_RECEIVE:
        G_t = G_t.reverse_slice_direction().reverse_time()
    if isinstance(G_t, TempFastGraph):
        M = calc_katz(G_t.get_causal_completion().switch_directions(), alpha=alpha).reshape(T, n)
        return M[0,:].copy().ravel()
    elif isinstance(G_t, SparseTempFastGraph):
        g_causal = G_t.get_sparse_causal_completion()
        g_causal_rev = g_causal.switch_directions()
        katz = calc_katz(g_causal_rev, alpha=alpha)

        from scipy.sparse import coo_array
        ts = g_causal.identifiers[:,1]
        ids =  g_causal.identifiers[:,0]
        print(coo_array((katz, (ts, ids)), shape = (T, n)).todense().T)


        return get_leftmost_entry_for_nodes(katz, g_causal.identifiers, n)
    else:
        raise NotImplementedError()





def calc_temp_katz_iter(G_temp, alpha=0.1, epsilon=1e-15, max_iter=100, beta=None, kind=KIND_BROADCAST):
    """Efficiently compute the Katz centrality from a sparse temporal graph

    alpha:
        the decay parameter
    epsilon:
        the numerical precision desired
    max_iter:
        the maximum number of iterations required
    beta:
        the starting vector for the centrality calculation
    kind:
        Either "broadcast" or "receive" to compute the broadcast or receive centrality

    This algorithm runs in O(n_iter * E + N).
    """
    assert isinstance(G_temp, SparseTempFastGraph)
    if kind == KIND_RECEIVE:
        G_temp = G_temp.reverse_time()
    elif kind==KIND_BROADCAST:
        G_temp = G_temp.reverse_slice_direction()
    if beta is None:
        beta=np.ones(G_temp.num_nodes)
    for G_t in reversed(G_temp.slices):
        beta_t = np.empty(G_t.num_nodes)
        for global_v, local_v in G_t.mapping.items():
            beta_t[local_v] = beta[global_v]

        katz = calc_katz_iter(G_t, alpha=alpha, epsilon=epsilon, max_iter=max_iter, beta=beta_t)

        for global_v, local_v in G_t.mapping.items():
            beta[global_v] = katz[local_v]
    return beta