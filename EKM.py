import numpy as np


def h(z, r):
    if r != 0:
        return z ** (1 / r)
    else:
        return np.exp(z)


def hinv(z, r):
    if r != 0:
        if r < 0 and z == 0:
            return float('inf')  # Infinity
        else:
            return z ** r
    else:
        if z != 0:
            return np.log(z)
        else:
            return float('-inf')  # Negative infinity


def wpmekml(x, w, r):
    """
    EKM algorithm for weighted power mean with r finite; left endpoint.
    x is an array of interval endpoints for x(i):  [xmin(i),xmax(i)]
    w is an array of interval endpoints for w(i):  [wmin(i),wmax(i)]
    """
    N = len(x)
    if r < 0:
        for xi in x[:, 0]:  # Assume x is a 2D array with interval endpoints
            if xi == 0:
                return 0

    # Sorting based on the first column of x combined with w
    aug = np.column_stack((x[:, 0], w))
    aug = aug[aug[:, 0].argsort()]  # Sorting by the first column
    xminsort, u, v = aug[:, 0], aug[:, 1], aug[:, 2]

    k = int(round(N / 2.4)) - 1
    a = sum(hinv(xminsort[i], r) * v[i] for i in range(k + 1)) + \
        sum(hinv(xminsort[i], r) * u[i] for i in range(k + 1, N))
    b = sum(v[:k + 1]) + sum(u[k + 1:])
    y = a / b
    yL = h(y, r)
    kL = k

    for kprime in range(N // 2):
        if xminsort[kprime] <= h(y, r) <= xminsort[kprime + 1]:
            break

    if xminsort[0] == xminsort[-1]:
        return xminsort[0]

    while kprime != k:
        s = np.sign(kprime - k)
        aprime = a + s * sum(hinv(xminsort[i], r) * (v[i] - u[i]) for i in range(min(k, kprime) + 1, max(k, kprime) + 1))
        bprime = b + s * sum(v[i] - u[i] for i in range(min(k, kprime) + 1, max(k, kprime) + 1))
        yprime = aprime / bprime
        y, a, b, k, yL, kL = yprime, aprime, bprime, kprime, h(yprime, r), kprime

        for kprime in range(N // 2):
            if xminsort[kprime] <= h(y, r) <= xminsort[kprime + 1]:
                break

    return yL


def wpmekmr(x, w, r):
    """
    EKM algorithm for weighted power mean with r finite; right endpoint.
    x is an array of interval endpoints for x(i):  [xmin(i),xmax(i)]
    w is an array of interval endpoints for w(i):  [wmin(i),wmax(i)]
    """
    N = len(x)
    aug = np.column_stack((x[:, 1], w))
    aug = aug[aug[:, 0].argsort()]  # Sorting by the first column
    xmaxsort, v, u = aug[:, 0], aug[:, 1], aug[:, 2]

    k = int(round(N / 1.7)) - 1
    a = sum(hinv(xmaxsort[i], r) * v[i] for i in range(k + 1)) + \
        sum(hinv(xmaxsort[i], r) * u[i] for i in range(k + 1, N))
    b = sum(v[:k + 1]) + sum(u[k + 1:])
    y = a / b
    yR = h(y, r)
    kR = k

    for kprime in range(N // 2):
        if xmaxsort[kprime] <= h(y, r) <= xmaxsort[kprime + 1]:
            break

    if xmaxsort[0] == xmaxsort[-1]:
        return xmaxsort[0]

    while kprime != k:
        s = np.sign(kprime - k)
        aprime = a + s * sum(hinv(xmaxsort[i], r) * (v[i] - u[i]) for i in range(min(k, kprime) + 1, max(k, kprime) + 1))
        bprime = b + s * sum(v[i] - u[i] for i in range(min(k, kprime) + 1, max(k, kprime) + 1))
        yprime = aprime / bprime
        y, a, b, k, yR, kR = yprime, aprime, bprime, kprime, h(yprime, r), kprime

        for kprime in range(N // 2):
            if xmaxsort[kprime] <= h(y, r) <= xmaxsort[kprime + 1]:
                break

    return yR


def wpminfendpts(x, r):
    if r == float('-inf'):
        return np.min(x[:, 0]), np.min(x[:, 1])
    elif r == float('inf'):
        return np.max(x[:, 0]), np.max(x[:, 1])


def t2wpm(x, w, r):
    if r == float('inf') or r == float('-inf'):
        return wpminfendpts(x, r)
    else:
        return wpmekml(x, w, r), wpmekmr(x, w, r)


def wpm(x, w, r):
    if 0 < abs(r) < float('inf'):
        xx = x ** r
        return (w * xx).sum() / w.sum() ** (1 / r)
    elif r == 0:
        return np.product(x ** (w / w.sum()))
    elif r == float('-inf'):
        return np.min(x)
    elif r == float('inf'):
        return np.max(x)
