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
    a = x[0]
    b = x[1]
    for i in range(0, len(x)):
        #remove rows with infeasable endpoints
        if ((x < a[i] < b[i] < x1) and (b[i] - a[i] < x1 - x0)):
            if y is None:
                y = [a[i], b[i]]
            else:
                y = np.vstack((y, [a[i], b[i]]))
    return y

