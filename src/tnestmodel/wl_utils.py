import numpy as np
from numba import njit


@njit(cache=True)
def _array_based_partitions_equivalent(p1, p2):
    max_val = p1.max()
    assert p1.min() >= 0
    lookup = np.full(max_val+1, -1,  dtype=p2.dtype)
    for c1, c2 in zip(p1, p2):
        if lookup[c1] == -1:
            lookup[c1]=c2
        else:
            if lookup[c1]==c2:
                continue
            else:
                return False, f"Color class {c1} has at least two matches ({lookup[c1]}, {c2})"
    return True, ""


def partitions_equivalent(p1, p2):
    """Checks whether the two paritions are equivalent

    Returns:
        is_equivalent : bool indicates whether they are equivalent
        message : str, message helping with what went wrong
    """
    if not len(p1)==len(p2):
        return False, f"size of partitions disagrees {len(p1)} != {len(p2)}"
    num_colors1=len(np.unique(p1))
    num_colors2=len(np.unique(p2))
    if not  num_colors1== num_colors2:
        return False, f"number of partitions does not match, {num_colors1} != {num_colors2}"
    if np.max(p1) < len(p1)*2:
        return _array_based_partitions_equivalent(np.array(p1, dtype=np.int64).ravel(), np.array(p2, dtype=np.int64).ravel())
    else:
        raise NotImplementedError

def assert_partitions_equivalent(p1, p2):
    """Raises an AssertionError when partitions p1 and p2 are not equivalent"""
    are_equivalent, msg = partitions_equivalent(p1, p2)
    if are_equivalent:
        return True
    else:
        assert False, msg
