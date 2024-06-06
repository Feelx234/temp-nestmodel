from collections import deque
import numpy as np
from numba import njit
from numba.typed import Dict # pylint: disable=no-name-in-module
from numba.core import types
from tnestmodel.temp_utils import switch_slice_directions

@njit(cache=True)
def count_number_of_active_nodes_per_node(E, num_nodes):
    """Computes the number of active nodes per non-temporal node
    assumes the times used in E are positive
    """
    last_t = -np.ones(num_nodes, dtype=np.int64)
    num_active_per_node = np.zeros(num_nodes, dtype=np.int64)
    for u,v,t in E:
        assert u != v
        if last_t[u]!=t:
            last_t[u]=t
            num_active_per_node[u]+=1

        if last_t[v]!=t:
            last_t[v]=t
            num_active_per_node[v]+=1
    return num_active_per_node

@njit(cache=True)
def convert_edges_to_active_edges(E, num_active_per_node, num_active_nodes, num_nodes):
    """Converts temporal edges into edges connecting the active nodes


    Returns : active_edges, times_for_active
    """
    last_t = -np.ones(num_nodes, dtype=np.int64)
    last_active_node = cumsum_from_zero(num_active_per_node, full=False) - 1
    #last_active_node = np.empty(len(num_active), dtype=int)
    #last_active_node[0]=-1
    #last_active_node[1:] = np.cumsum(num_active)[:-1]-1

    E_active = np.empty((len(E), 2), dtype = np.int64)
    times_for_active = np.empty(num_active_nodes, dtype=np.int64)

    for i, (u,v,t) in enumerate(E):
        if last_t[u]!=t:
            last_t[u] = t
            last_active_node[u]+=1
            times_for_active[last_active_node[u]]=t
        new_u = last_active_node[u]

        if last_t[v]!=t:
            last_t[v] = t
            last_active_node[v]+=1
            times_for_active[last_active_node[v]]=t
        new_v = last_active_node[v]
        E_active[i,0] = new_u
        E_active[i,1] = new_v
    return E_active, times_for_active


MAX_HASH_BITS=64


def get_random_hashes(n, max_sum_length, seed=0):
    """Computes n random hashes

    Sums of hashes are correctly represented using at most max_sum_length"""
    def bit_length(x):
        return (int(x)-1).bit_length()
    bits_reserved_for_sum = bit_length(max_sum_length)
    max_value_hash = np.power(2, MAX_HASH_BITS - bits_reserved_for_sum, dtype=np.uint64) - np.uint64(1)
    rng = np.random.default_rng(seed)
    return rng.integers(0, max_value_hash, size=n, dtype=np.uint64, endpoint=True)

def compute_d_rounds(E : np.ndarray, num_nodes : int, d : int, h : int=-1, seed : int=0, reverse_slices=True):
    """Computes temporal wl of temporal edges E
    E : temporal edges, size = (num_edges, 3); u,v,t = E[0,:] represents a directed edge u->v at time t
    num_nodes: the number of non-temporal nodes
    d: maximum number of rounds to do the wl-algorithm; d<0 means compute until convergence
    h: absolute horizon of temporal nodes, i.e. an edge at time t can see nodes up to and including t+h; h<0 means infinite horizon
    seed: seed used to initialize pseudo random number generator generating used hash values
    """
    _, colors_per_round, _, _, num_active_per_node, E, _, times_for_active = _compute_d_rounds(
        E=E,
        num_nodes=num_nodes,
        d = d,
        h=h,
        seed=seed,
        reverse_slices=reverse_slices
    )
    return colors_per_round, repeat_active_nodes(num_active_per_node), times_for_active

def _compute_d_rounds(E : np.ndarray, num_nodes : int, d : int, h : int=-1, seed : int=0, reverse_slices=True):
    """Computes temporal wl of temporal edges E
    E : temporal edges, size = (num_edges, 3); u,v,t = E[0,:] represents a directed edge u->v at time t
    num_nodes: the number of non-temporal nodes
    d: maximum number of rounds to do the wl-algorithm; d<0 means compute until convergence
    h: absolute horizon of temporal nodes, i.e. an edge at time t can see nodes up to and including t+h; h<0 means infinite horizon
    seed: seed used to initialize pseudo random number generator generating used hash values
    """
    if reverse_slices:
        E = switch_slice_directions(E)
    assert d != 0
    order_in_time = np.argsort(E[:, 2])
    smallest_time = E[order_in_time[0],2]
    assert smallest_time >=0, "negative times currently not supported"
    E = E[order_in_time, :]

    num_active_per_node = count_number_of_active_nodes_per_node(E, num_nodes=num_nodes)
    total_active_nodes = num_active_per_node.sum()
    max_num_active = num_active_per_node.max() # maximum degree of a node, necessary to limit hash sizes

    E_active, times_for_active = convert_edges_to_active_edges(E, num_active_per_node, total_active_nodes, num_nodes)


    hashes = get_random_hashes(2 * total_active_nodes, max_sum_length = 2*max_num_active, seed=seed)
    hashes[total_active_nodes] = 0  # assign degree zero the zero hash

    colors = np.zeros(total_active_nodes, dtype=np.uint64)
    colors_per_round = [colors]
    cumsum_hashes_per_round = []

    if d > 0:
        max_num_iterations = d
    else:
        max_num_iterations  = total_active_nodes
    num_prev_colors = len(np.unique(colors))
    for _ in range(max_num_iterations):
        new_colors, cumsum_hashes = one_round(hashes, colors_per_round[-1], num_prev_colors, E_active, num_active_per_node, total_active_nodes, times_for_active, h)
        # print(new_colors)
        max_colors = new_colors.max()+1
        # print("max", max_colors, num_prev_colors)
        cumsum_hashes_per_round.append(cumsum_hashes)
        if max_colors==num_prev_colors: # stable colors reached, terminate
            break
        num_prev_colors = max_colors
        colors_per_round.append(new_colors)



    return h, colors_per_round, cumsum_hashes_per_round, hashes, num_active_per_node, E, E_active, times_for_active

@njit(cache=True)
def determine_potentially_changed_nodes(t, h, left_e_index, right_e_index, E, E_active, changed_nodes, changed_indicator, hashes, cumsum_hashes, prev_colors):
    """Computes potentially changed nodes and updates hashes"""
    num_changed = 0
    # process edges no longer visible
    #  we find all nodes that loose an edge by increasing the time
    #  for these we increase the left active node counter which
    #  will subsequently affect that particular nodes hash

    if h < 0 and t == E[0,2]:
        h = E[len(E)-1, 2] - E[0,2] +1

    # process newly visible edges
    #  by advancing the time, we add newly visible edges
    while right_e_index < len(E):
        (u_act, _) = E_active[right_e_index,:]
        (_, v, t_e) = E[right_e_index,:]
        # print(t, "B   ", u_act, v, t_e)
        if t_e > t + h:
            # print("skipped")
            # we no longer see this edge or any future edges, break
            break
        #self.right_active_node[v]+=1
        hash_to_add = hashes[prev_colors[u_act]]
        cumsum_hashes[v]+=hash_to_add
        if not changed_indicator[v]:
            changed_nodes[num_changed]=v
            changed_indicator[v]=True
            num_changed+=1
        right_e_index+=1
    while left_e_index < len(E):
        (u_act, _) = E_active[left_e_index,:]
        (_, v, t_e) = E[left_e_index,:]

        # print(t, "A   ", u_act, v, t_e)
        if t_e >= t:
            # print("skipped")
            # we arrived at edges that can be potentially seen
            break

        #self.left_active_node[v]+=1
        hash_to_add = hashes[prev_colors[u_act]]
        cumsum_hashes[v]-=hash_to_add
        if not changed_indicator[v]:
            changed_nodes[num_changed]=v
            changed_indicator[v]=True
            num_changed+=1
        left_e_index+=1
    # reset the changed indicator
    for i in range(num_changed):
        v = changed_nodes[i]
        changed_indicator[v]=False
    return num_changed, left_e_index, right_e_index

@njit(cache=True)
def _advance_time(t, h, left_e_index, right_e_index, E, E_active, prev_colors, hashes, cumsum_hashes, changed_nodes, changed_indicator, current_colors, hash_dict):
    """Computes the colors of all nodes for time t

    This function runs in O(E)

    It returns the name of nodes whose color changed during this update
    """

    num_changed, left_e_index, right_e_index = determine_potentially_changed_nodes(t, h, left_e_index, right_e_index, E, E_active, changed_nodes, changed_indicator, hashes, cumsum_hashes, prev_colors)
    for i in range(num_changed):
        v = changed_nodes[i]
        if  cumsum_hashes[v]==0: # use "special color zero for degree zero nodes"
            current_colors[v] = 0 # this node does not see any edges
            continue
        the_hash = cumsum_hashes[v]

        if the_hash in hash_dict:
            current_colors[v] = hash_dict[the_hash]
        else:
            # uses globally valid colors, size of hash dict needs to grow
            new_color = len(hash_dict)+1 # color zero reserved for degree zero nodes
            current_colors[v] = new_color
            hash_dict[the_hash]= new_color
    return changed_nodes[:num_changed], left_e_index, right_e_index


def advance_time_non_compiled(self, t):
    """Computes the colors of all nodes for time t

    This function runs in O(E)

    It returns the name of nodes whose color changed during this update
    """

    if self.d == 0:
        return set()
    self.t = t
    changed_nodes = set()

    # process edges no longer visible
    #  we find all nodes that loose an edge by increasing the time
    #  for these we increase the left active node counter which
    #  will subsequently affect that particular nodes hash
    while self.left_e_index < len(self.E):
        (u_act, _) = self.E_active[self.left_e_index,:]
        (_, v, t_e) = self.E[self.left_e_index,:]

        # print(t, "A   ", u_act, v, t_e)
        if t_e >= t:
            # print("skipped")
            # we arrived at edges that can be potentially seen
            break

        #self.left_active_node[v]+=1
        hash_to_add = self.hashes[self.prev_colors[u_act]]
        self.cumsum_hashes[v]-=hash_to_add
        changed_nodes.add(v)
        self.left_e_index+=1
    np.seterr(all='warn')
    # process newly visible edges
    #  by advancing the time, we add newly visible edges
    while self.right_e_index < len(self.E):
        (u_act, _) = self.E_active[self.right_e_index,:]
        (_, v, t_e) = self.E[self.right_e_index,:]
        # print(t, "B   ", u_act, v, t_e)
        if t_e > t + self.h:
            # print("skipped")
            # we no longer see this edge or any future edges, break
            break
        #self.right_active_node[v]+=1
        hash_to_add = self.hashes[self.prev_colors[u_act]]
        self.cumsum_hashes[v]+=hash_to_add
        changed_nodes.add(v)
        self.right_e_index+=1
    changed_nodes = np.fromiter(changed_nodes, count = len(changed_nodes), dtype=np.int64)
    #changed_nodes.sort()

    for v in changed_nodes:
        if  self.cumsum_hashes[v]==0:
            self.current_colors[v] = 0 # this node does not see any edges
            continue
        the_hash = self.cumsum_hashes[v]

        if self.mode=="global":
            # uses globally valid colors, size of hash dict needs to grow
            if the_hash in self.hash_dict:
                self.current_colors[v] = self.hash_dict[the_hash]
            else:
                new_color = len(self.hash_dict)+1 # color zero reserved for degree zero nodes
                self.current_colors[v] = new_color
                self.hash_dict[the_hash]= new_color
        elif self.mode == "local":
            # in this mode we only keep only those hashes that are relevant in this slice
            # the purpose of this is to keep the memory requirement to O(V+E)

            # decrease count of previous color
            #  if this color has no remaining uses, free this color for future uses
            last_color = self.current_colors[v]
            self.color_count_arr[last_color] -=1
            if self.color_count_arr[last_color] == 0:
                self.colorqueue.appendleft(last_color)
                del self.hash_dict[self.current_hashes[v]]
            self.current_hashes[v] = the_hash


            if the_hash in self.hash_dict:
                # use the color associated with this hash
                self.current_colors[v] = self.hash_dict[the_hash]
            else:
                # use a new color and make it available to other nodes by storing
                #  it in the hash_dict
                new_color = self.colorqueue.popleft()
                self.current_colors[v] = new_color
                self.hash_dict[the_hash]= new_color
            # make sure the number of nodes with this color is correct
            self.color_count_arr[self.current_colors[v]] +=1
        else:
            assert False, "This should not be reached"
    return changed_nodes



class TemporalColorsStruct:
    """A structure able to provide the current color of all nodes

    Runtime:
        When iterating increasing in time, this provides the current colors of all nodes in O(E+T)
    """
    def __init__(self, h, colors_per_round, cumsum_hashes_per_round, hashes, num_active_per_node, E, E_active, times_for_active):
        self.colors_per_round=colors_per_round
        self.cumsum_hashes_per_round=cumsum_hashes_per_round
        self.hashes = hashes
        self.num_active_per_node = num_active_per_node
        assert len(E)==len(E_active)
        self.E = E
        self.E_active = E_active
        self.times_for_active = times_for_active

        self.colors_for_active = None

        self.d = None
        self.h = h
        self.t = None
        self.base_active_node = cumsum_from_zero(self.num_active_per_node, full=True)
        self.num_nodes = len(num_active_per_node)

        self.prev_colors = None
        self.cumsum_hashes = None

        self.current_colors = np.zeros(self.num_nodes, dtype=np.int64)
        self.current_hashes = np.zeros(self.num_nodes, dtype=np.uint64)
        self.changed_nodes = np.empty(self.num_nodes, dtype=np.int64)
        self.changed_indicator = np.zeros(self.num_nodes, dtype=np.int64) # needs to be zeros to work properly
        self.left_e_index = None
        self.right_e_index = None
        self.hash_dict = Dict.empty(
            key_type=types.uint64,
            value_type=types.int64,
        )
        self.mode="global"
        self.color_count_arr = np.zeros(self.num_nodes, dtype=np.int64)
        self.colorqueue = None

    def reset_colors(self, d, mode=None):
        """Resets colors and allows to restart the coloring process"""
        self.d = d
        # assert h > 0
        self.t = -1
        if d > 0:
            self.prev_colors = self.colors_per_round[d-1]
            self.cumsum_hashes = np.zeros_like(self.cumsum_hashes_per_round[d-1])
        if not mode is None:
            self.mode=mode
        self.current_colors[:]=0
        self.current_hashes[:]=0
        self.left_e_index = 0
        self.right_e_index = 0
        self.hash_dict.clear()
        self.color_count_arr[:]=0
        #if self.mode != "global":
        self.colorqueue = deque(range(1,self.num_nodes+1)) # color zero reserved for degree 0 nodes



    def advance_time(self, t):
        """Computes the colors of all nodes for time t

        This function runs in O(E)

        It returns the name of nodes whose color changed during this update
        """

        assert self.t < t, "Can only move forward in time, if you want to go to an earlier time use .reset_colors again"
        self.t=t
        if self.d == 0:
            return np.empty(0, dtype=np.int64)
        if self.mode=="local":
            return advance_time_non_compiled(self, t)
        changed_nodes, self.left_e_index, self.right_e_index = _advance_time(t,
                                                                             self.h,
                                                                             self.left_e_index,
                                                                             self.right_e_index,
                                                                             self.E,
                                                                             self.E_active,
                                                                             self.prev_colors,
                                                                             self.hashes,
                                                                             self.cumsum_hashes,
                                                                             self.changed_nodes,
                                                                             self.changed_indicator,
                                                                             self.current_colors,
                                                                             self.hash_dict)
        return changed_nodes

    def get_colors_all_times(self, d, mode=None):
        """Computes the colors for all times
        this runs in O(V*T + E)
        """
        self.reset_colors(d, mode=mode)
        times = np.unique(self.E[:,2]).ravel()
        colors = []
        for t in times:
            self.advance_time(t)
            colors.append(self.current_colors.copy())
        return np.hstack(colors)





@njit(cache=True)
def one_round(hashes : np.ndarray, colors : np.ndarray, num_colors : int, E_active : np.ndarray, num_active_per_node : np.ndarray, num_active_nodes : int, times : np.ndarray, h : int):
    """Performs one round of wl color refinement
    hashes : an array of hashes of length num_active_nodes + num_colors
    colors : an array indicating color partitions. active nodes with the same color have number; numbers in range [0, num_active_nodes]
    num_colors : the number of different colors
    E_active : the number of edges connecting active nodes
    num_active_per_node : number of active nodes per non-temporal node
    num_active_nodes : the total number of active nodes
    times : the times assigned to active nodes
    h : the absolute horizont of nodes

    Returns : array of colors  representing the partitions after another round of wl refinement
    """

    simple_hashes = np.zeros(num_active_nodes, dtype=np.uint64)
    for u,v in E_active:
        simple_hashes[v]+= hashes[colors[u]]

    #print(simple_hashes)
    # agg hashes according to degrees
    n = 0
    for d in num_active_per_node:
        if d > 1:
            for i in range(n+d-2, n-1, -1):
                simple_hashes[i]+=simple_hashes[i+1]
        n+=d

    cumsum_hashes = simple_hashes
    cumsum_hashes_out = cumsum_hashes.copy()
    if h < 0:
        agg_hashes = cumsum_hashes
    else:
        agg_hashes = adjust_hashes_for_finite_horizon(cumsum_hashes, num_active_per_node, times, h)
    #print(agg_hashes)
    ## if external colors are available
    #if num_colors > 1:
    #    for i, c in enumerate(colors):
    #        agg_hashes[i]+=hashes[c+np.uint64(len(colors))]
    # sort hashes, such that similar hashes are adjacent
    order = np.argsort(agg_hashes)

    # find colors (i.e. numbers from 0 to number of nodes)
    #    to replace hashes
    current_color = 0
    current_hash = agg_hashes[order[0]]
    out_colors = np.empty(num_active_nodes, dtype=np.int64)
    for i in order:
        if current_hash == agg_hashes[i]:
            out_colors[i] = current_color
        else:
            current_color += 1
            current_hash = agg_hashes[i]
            out_colors[i] = current_color
    return out_colors, cumsum_hashes_out


@njit(cache=True)
def adjust_hashes_for_finite_horizon(cumsum_hashes, num_active_per_node, times, h):
    """Computes hashes per active node for temporal graph with finite horizon
    cumsum_hashes : Cummulatives hashes per node
    num_active_per_node : count of the number of active node per node
    times : time corresponding to each of the active nodes
    h : finite absolute horizon, a node can see edges that are at most h timesteps in the future
    """
    hashes_out = cumsum_hashes.copy()
    n = 0
    for d in num_active_per_node:
        if d > 1:
            last_unseen_pointer = n+d
            hash_to_subtract = np.uint64(0) # do not subtract anything, need np.uint64 otherwise wrong result!
            last_time_seeable = times[n+d-1] # one time more than the latest time of this node
            for i in range(n+d-1, n-1, -1):
                curr_time = times[i]
                curr_max_time_seeable = curr_time + h
                #print("times", curr_max_time_seeable, last_time_seeable)
                if curr_max_time_seeable >= last_time_seeable:
                    # can see everything that is subtracted
                    pass
                else:
                    # adjust the pointer to the latest non seeable time
                    last_unseen_pointer-=1
                    while last_unseen_pointer > i  and curr_max_time_seeable < times[last_unseen_pointer]:
                        last_unseen_pointer-=1
                    last_unseen_pointer += 1
                    hash_to_subtract = cumsum_hashes[last_unseen_pointer]
                    last_time_seeable = times[last_unseen_pointer-1]
                #print(i, hashes_out[i], hash_to_subtract, hash_to_subtract<hashes_out[i], last_unseen_pointer)

                hashes_out[i]-=hash_to_subtract
                #print(hashes_out[i])
        n+=d
    #print(hashes_out)
    return hashes_out

@njit(cache=True)
def cumsum_from_zero(vals, full=True):
    """Returns a cumsum of vals starting from zero
    Example:
    input: [2,4,5]
    output: [0,2,6,11]

    if full=False, removes the last element
    """
    if full:
        out = np.empty(len(vals)+1, dtype=np.int64)
        out[0]=0
        out[1:] = np.cumsum(vals)
    else:
        out = np.empty(len(vals), dtype=np.int64)
        out[0]=0
        out[1:] = np.cumsum(vals)[:-1]
    return out


@njit(cache=True)
def repeat_active_nodes(num_active_per_node):
    """ Repeats the active nodes by the number of the nodes
    Input: [1,4,3]
    Output [0,1,1,1,1,2,2,2]
    """
    length = np.sum(num_active_per_node)
    out = np.empty(length, dtype=np.int64)
    n=0
    for i, d in enumerate(num_active_per_node):
        out[n:n+d]=i
        n+=d
    return out
