def list_compare(list_a: [], list_b: []) -> bool:
    if len(list_a) != len(list_b):
        return False
    comp = [list_a[i].__eq__(list_b[i]) for i in range(len(list_a))]
    return False not in comp