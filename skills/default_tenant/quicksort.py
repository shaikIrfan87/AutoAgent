def quicksort(lst):
    def _y_comb(f, max_depth=500):
        depth = 0
        def fix(*args):
            nonlocal depth
            if depth >= max_depth: raise RecursionError('Y recursion limit')
            depth += 1
            try: return f(fix)(*args)
            finally: depth -= 1
        return fix
    return _y_comb(lambda rec: lambda lst: lst if len(lst) <= 1 else rec([x for x in lst[1:] if x <= lst[0]]) + [lst[0]] + rec([x for x in lst[1:] if x > lst[0]]))(lst)
