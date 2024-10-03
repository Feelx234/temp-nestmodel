from collections import defaultdict, Counter
import numpy as np
from numba.typed import Dict #pylint: disable=no-name-in-module
from numba import njit
from numpy.testing import assert_array_equal















from nestmodel.graph_properties import _source_only_number_of_flips_possible, _normal_number_of_flips_possible

@njit(cache=True)
def update_color_counts(affected_nodes, last_colors, new_colors, number_of_nodes_per_color):
    for node in affected_nodes:
        old_color = last_colors[node]
        new_color = new_colors[node]
        if old_color==new_color:
            continue
        # remove old color from count
        if number_of_nodes_per_color[old_color] == 1:
            del number_of_nodes_per_color[old_color]
        else:
            number_of_nodes_per_color[old_color]-=1
        # add new color to count
        if new_color in number_of_nodes_per_color:
            number_of_nodes_per_color[new_color]+=1
        else:
            number_of_nodes_per_color[new_color]=1
        last_colors[node] = new_color



class NumberOfFlipsCalculator():
    def __init__(self, G_temp, h, s = None):
        self.G_temp = G_temp
        self.h = h
        if s is None:
            self.struct = G_temp.s#get_temporal_wl_struct(h=h, base_edges=True)
        else:
            self.struct = s
        if G_temp.is_directed:
            self.last_colors = np.zeros(G_temp.num_nodes, dtype=np.int64) # assumes degree zero nodes have color zero!
            self.number_of_nodes_per_color = Dict()
            self.number_of_nodes_per_color[0] = G_temp.num_nodes
        else:
            self.last_colors = None
            self.number_of_nodes_per_color = None


        self.d = None

    def prepare(self, d):
        self.d = d
        self.struct.reset_colors(d=d, mode="global")
        if self.G_temp.is_directed:
            self.last_colors[:]=0
            self.number_of_nodes_per_color.clear()
            self.number_of_nodes_per_color[0] = self.G_temp.num_nodes

    def calc_for_slice(self, G, t):
        if self.G_temp.is_directed:
            affected_nodes = self.struct.advance_time(t)
            #print(G.global_edges, affected_nodes)
            affected_nodes = np.fromiter(affected_nodes, count = len(affected_nodes), dtype=np.int64)
            #print("current_colors", self.struct.current_colors[G.global_nodes])
            #print("set     colors", G.base_partitions[self.d])
            assert_array_equal(self.struct.current_colors[G.global_nodes], G.base_partitions[self.d])
            update_color_counts(affected_nodes, self.last_colors, self.struct.current_colors, self.number_of_nodes_per_color)
            return _source_only_number_of_flips_possible(G, self.d, self.number_of_nodes_per_color)
        else:
            return _normal_number_of_flips_possible(G, d)















#  from collections import defaultdict, Counter
def _add_nodes_to_maps(u,v, basic, counter, is_directed):
    if is_directed:
        basic[u].add(v)
        counter[u][v]+=1
    else:
        basic[u].add(v)
        basic[v].add(u)
        counter[u][v]+=1
        counter[v][u]+=1

def _remove_nodes_from_maps(u,v,basic, counter, is_directed):
    if is_directed:
        counter[u][v]-=1
        if counter[u][v]==0:
            basic[u].remove(v)
    else:
        counter[u][v]-=1
        if counter[u][v]==0:
            basic[u].remove(v)
        counter[v][u]-=1
        if counter[v][u]==0:
            basic[v].remove(u)


class NumberOfTrianglesCalculator():
    """Computes the number of forward orinted triangles

    I.e. counts the number of tuples ( (u,v,t1), (v, w, t2), (w, u, t3) ) such that t1<t2<t3
    If strict=False, the strict inequalities become inclusive inequalities (i.e. <=).

    """
    def __init__(self, G_temp, h=None, strict=True):
        self.G_temp = G_temp
        self.edges = G_temp.to_temporal_edges()
        if not h is None:
            raise NotImplementedError("Currently only infinite horizon is supported")
        self.count=0
        self.strict = strict # set to True to only count triangles where edges are strict forward in time
        self.future_nodes = defaultdict(set)
        self.future_nodes_count = defaultdict(Counter)
        self.past_nodes = defaultdict(set)
        self.past_nodes_count = defaultdict(Counter)

    def prepare(self):
        """Initialize the calculator"""
        self.future_nodes.clear()
        self.future_nodes_count.clear()
        self.past_nodes.clear()
        self.past_nodes_count.clear()
        for u,v,_ in self.edges:
            _add_nodes_to_maps(u,v, self.future_nodes, self.future_nodes_count, self.G_temp.is_directed)

    def calc_for_slice(self, G, _):
        count = 0
        E = G.global_edges.copy()
        for u,v in E:
            if self.strict:
                _remove_nodes_from_maps(u,v, self.future_nodes, self.future_nodes_count, self.G_temp.is_directed)
            else:
                _add_nodes_to_maps(v,u, self.past_nodes, self.past_nodes_count, self.G_temp.is_directed) # v, u is correct
        # print(self.past_nodes)
        # print(self.future_nodes)

        for u,v in E:
            possible_third_nodes = self.past_nodes[u].intersection(self.future_nodes[v])
            # print(u,v, possible_third_nodes)
            for third_node in possible_third_nodes:
                # print(u,v,t, third_node, self.past_nodes_count[u][third_node]*self.future_nodes_count[v][third_node])
                count+=self.past_nodes_count[u][third_node]*self.future_nodes_count[v][third_node]
        for u,v in E:
            if self.strict:
                _add_nodes_to_maps(v,u, self.past_nodes, self.past_nodes_count, self.G_temp.is_directed)  # v, u is correct
            else:
                _remove_nodes_from_maps(u,v, self.future_nodes, self.future_nodes_count, self.G_temp.is_directed)
        return count





class EdgePersistenceCalculator():
    def __init__(self, G_temp, h=None):
        self.G_temp = G_temp
        self.edges = G_temp.to_temporal_edges()
        if not h is None:
            raise NotImplementedError()
        self.count=0
        self.previous_edges = set()
        self.previous_out_degrees = defaultdict(int)
        self.previous_in_degrees = defaultdict(int)


    def prepare(self):
        self.previous_edges.clear()
        self.previous_out_degrees.clear()
        self.previous_in_degrees.clear()

    def calc_for_slice(self, G, _):
        new_edges = set()
        out_degrees = defaultdict(int)
        in_degrees = defaultdict(int)
        for v,u in G.global_edges:
            if self.G_temp.is_directed:
                new_edges.add((u,v))
                out_degrees[u]+=1
                in_degrees[v]+=1
            else:
                new_edges.add((u,v))
                new_edges.add((v,u))
                out_degrees[u]+=1
                out_degrees[v]+=1
                in_degrees[u]+=1
                in_degrees[v]+=1


        shared_edges = self.previous_edges.intersection(new_edges)
        value=0
        if len(shared_edges)>0:
            shared_degree = defaultdict(int)
            for u,v in shared_edges:
                if self.G_temp.is_directed:
                    shared_degree[u]+=1
                else:
                    shared_degree[u]+=1
            shared_nodes = set(u for u,v in shared_edges)

            value = 0
            for node in shared_nodes:
                value += shared_degree[node] / (np.sqrt(self.previous_out_degrees[node])*np.sqrt(out_degrees[node]))
        self.previous_out_degrees = out_degrees
        self.previous_in_degrees = in_degrees
        self.previous_edges = new_edges

        return value





def get_aggregated_graph(G):
    m = defaultdict(int)
    if G.is_directed:
        for u,v,_ in G.to_temporal_edges():
            m[(u,v)]+=1
    else:
        for u,v,_ in G.to_temporal_edges():
            m[(u,v)]+=1
            m[(v,u)]+=1
    return m


def _get_aggregated_graph(G, ignore_directionality=False):
    E = G.to_temporal_edges()
    return _get_aggregated_graph_from_edges(E, ignore_directionality or G.is_directed)

@njit(cache=True)
def _get_aggregated_graph_from_edges(E, is_directed):
    m = Dict()
    if is_directed:
        for i in range(E.shape[0]):
            u = E[i,0]
            v = E[i,1]
            if (u,v) not in m:
                m[(u,v)] = 1
            else:
                m[(u,v)]+=1
    else:
        for i in range(E.shape[0]):
            u = E[i,0]
            v = E[i,1]
            if (u,v) not in m:
                m[(u,v)] = 1
            else:
                m[(u,v)]+=1
            if (v,u) not in m:
                m[(v,u)] = 1
            else:
                m[(v,u)]+=1
    return m

@njit(cache=True)
def _count_triangles_in_aggregated(m, succ):
    c = 0
    for (u,v), mul in m.items():
        for w in succ[v]:
            if (w,u) in m:
                c+=mul*m[(w,u)]*m[(v,w)]
    return c


@njit(cache=True)
def _count_3paths_in_aggregated(m, num_nodes):
    succ = np.zeros(num_nodes, dtype=np.int64)
    pred = np.zeros(num_nodes, dtype=np.int64)
    for (u,v), mul in m.items():
        succ[v]+=mul
        pred[u]+=mul

    c = 0
    for (u,v), mul in m.items():
        c+= mul * pred[u] * succ[v]
    return c


@njit(cache=True)
def _get_successors_from_aggregated(m):
    succ = Dict()
    for (u,v) in m.keys():
        if u in succ:
            succ[u][v] = 1
        else:
            tmp = Dict()
            tmp[v]=1
            succ[u]=tmp
    return succ

def count_triangles_in_aggregated(m, G):
    c = 0
    for (u,v), mul in m.items():
        for w in range(G.num_nodes):
            if (w,u) in m and (v,w) in m:
                c+=mul*m[(w,u)]*m[(v,w)]
    return c


def get_all_triangles(G):
    m = get_aggregated_graph(G)
    c = count_triangles_in_aggregated(m, G)
    return c

def numba_get_all_triangles(G):
    m = _get_aggregated_graph(G)
    c = _count_3paths_in_aggregated(m, G.num_nodes)
    return c


def numba_get_3paths_triangles(G):
    m = _get_aggregated_graph(G)
    succ = _get_successors_from_aggregated(m)
    c = _count_triangles_in_aggregated(m, succ)
    return c

def _number_of_edges(G):
    return len(G.edges)

def get_total_number_of_edges(G):
    _, arr = G.compute_for_each_slice(_number_of_edges, min_size=1, call_with_time=False, dtype=np.int64)
    return arr.sum()


def get_edge_persistence(G):
    calculator = EdgePersistenceCalculator(G)
    calculator.prepare()
    _, arr = G.compute_for_each_slice(calculator.calc_for_slice, min_size=1, call_with_time=True)
    return arr.sum()



def get_burstiness_from_taus(taus):
    m = np.mean(taus)
    s = np.std(taus)
    return (s-m)/(s+m)

def burstiness(G):
    in_taus, out_taus, taus = inter_arrival_times(G)

    return (get_burstiness_from_taus(in_taus),
            get_burstiness_from_taus(out_taus),
            get_burstiness_from_taus(taus))



def inter_arrival_times(G):

    in_taus = []
    out_taus = []
    taus = []
    in_active = {}
    out_active = {}
    active = {}
    for u in range(G.num_nodes):
        in_active[u] = None
        out_active[u] = None
        active[u] = None
    for u,v,t in G.to_temporal_edges():
        if in_active[v] is not None:
            in_taus.append(t-in_active[v])
        if out_active[u] is not None:
            out_taus.append(t-out_active[u])
        if active[u]:
            taus.append(t-active[u])
        if active[v]:
            taus.append(t-active[v])
        active[u]=t
        active[v]=t
        out_active[u]=t
        in_active[v]=t
    if G.is_directed:
        return np.array(in_taus), np.array(out_taus), np.array(taus)
    else:
        return np.array(taus), np.array(taus), np.array(taus)



def causal_triangles(G):
    calculator = NumberOfTrianglesCalculator(G, strict=True)
    calculator.prepare()
    _, arr = G.compute_for_each_slice(calculator.calc_for_slice, min_size=1, call_with_time=True, dtype=int)
    return arr.sum()