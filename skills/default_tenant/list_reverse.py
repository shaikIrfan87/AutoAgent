def reverse_fn(lst):
    def _y_comb(f, max_depth=500):
        depth = 0
        def fix(*args):
            nonlocal depth
            if depth >= max_depth: raise RecursionError('Y recursion limit')
            depth += 1
            try: return f(fix)(*args)
            finally: depth -= 1
        return fix
    return _y_comb(lambda rec: lambda lst: [] if not lst else rec(lst[1:]) + [lst[0]])(lst)
