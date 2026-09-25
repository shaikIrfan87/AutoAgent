def flatten(matrix):
    res = []
    for row in matrix:
        for x in row:
            res.append(x)
    return res