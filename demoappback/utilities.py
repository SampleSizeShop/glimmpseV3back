import numpy as np


def list_compare(list_a: [], list_b: []) -> bool:
    if len(list_a) != len(list_b):
        return False
    comp = [list_a[i].__eq__(list_b[i]) for i in range(len(list_a))]
    return False not in comp

def kronecker_list(l: []):
    if not l or len(l) == 0:
        return None
    newlist = l[::-1]
    prod = newlist.pop()
    while len(newlist) > 0:
        prod = np.kron(prod, newlist.pop())
    return prod

def serialise_matrix(m):
    if isinstance(m, np.matrix):
        ret = m.tolist()
        return ret
    else:
        return None