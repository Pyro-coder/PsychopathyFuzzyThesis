import numpy as np


def trap(x, ht, lb, lt, rt, rb):
    #trapezoid of height ht, left base lb, left top lt, right top rt, right base rb
    if (x < lb):
        return 0
    if (x > rb):
        return 0
    if (lt <= x <= rt):
        return ht
    if (lb <= x < lt):
        tr = ht * (x - lb) / (lt - lb)
    if (rt < x <= rb):
        tr = ht * (rb - x) / (rb - rt)
    return tr


def bad_data(x, x0, x1):
    #eliminates invalid intervals
    y = None
    for i in range(0, len(x)):
        a, b = x[i]
        #remove rows with infeasible endpoints
        if ((x0 < a < b < x1) and (b - a < x1 - x0)):
            if y is None:
                y = [[a, b]]
            else:
                y.append([a, b])
    return y

data = [[0, 10],
        [-1, 10],
        [0, 11],
        [5, 4],
        [1, 9],
        [3, 3],
        [50, 10]]


print(bad_data(data, 0, 10))

