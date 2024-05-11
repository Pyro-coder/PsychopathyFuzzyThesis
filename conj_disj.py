import EKM

def gcd(x, w, r):
    """Generalized type-2 conjunction/disjunction operator using weighted power mean
    x and w are nx2 arrays containing interval endpoints; r is the exponent of the WPM
    If r > 0, we use the De Morgan dual operator
    """
    if r <= 0:
        out = EKM.t2wpm(x, w, r)