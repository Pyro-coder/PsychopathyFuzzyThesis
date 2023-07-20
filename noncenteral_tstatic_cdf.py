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