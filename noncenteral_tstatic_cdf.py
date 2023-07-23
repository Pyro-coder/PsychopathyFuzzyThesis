import math

import numpy as np
import openpyxl
from scipy.integrate import quad
from scipy.optimize import root
from scipy.stats import chi2, norm


def trap(x, ht, lb, lt, rt, rb):
    # trapezoid of height ht, left base lb, left top lt, right top rt, right base rb
    global tr
    if x < lb:
        return 0
    if x > rb:
        return 0
    if lt <= x <= rt:
        return ht
    if lb <= x < lt:
        tr = ht * (x - lb) / (lt - lb)
    if rt < x <= rb:
        tr = ht * (rb - x) / (rb - rt)
    return tr


def bad_data(x, x0, x1):
    # eliminates invalid intervals
    y = None
    for i in range(0, len(x)):
        a, b = x[i]
        # remove rows with infeasible endpoints
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

web_data_copy = openpyxl.load_workbook('excel\WEBdatacopy.xlsx')

wdc_sheet = web_data_copy['data']

range_values = wdc_sheet['AI2:AJ175']


# for row in range_values:
#    for cell in row:
#        print(cell.value, end=',')
#    print()

def box_and_whisker(x):
    # calculate quartiles and inter-quartile ranges on right or left interval endpoints
    # x is an n-vector of right or left interval endpoints
    # calculate quartile index
    global q75, q25
    nq = np.floor(len(x) / 4)

    # skip this test if nq = 0
    if (nq == 0):
        return x

    # find first and third quartile values
    remainder = np.mod(len(x), 4)
    y = np.sort(x)
    if (remainder == 0):
        q25 = (y[int(nq)] + y[int(nq) - 1]) / 2
        q75 = (y[int(3 * nq)] + y[int(3 * nq) - 1]) / 2
    elif (remainder == 1):
        q25 = (y[int(nq)] + y[int(nq) - 1]) / 2
        q75 = (y[int(3 * nq)] + y[int(3 * nq + 1)]) / 2
    elif (remainder == 2):
        q25 = y[int(nq)]
        q75 = y[int(3 * nq + 1)]
    elif (remainder == 3):
        q25 = y[int(nq)]
        q75 = y[int(3 * nq + 2)]

    # find inner quartile range and bounds of valid endpoints
    iqr = q75 - q25
    xmin = q25 - 1.5 * iqr
    xmax = q75 + 1.5 * iqr
    return [xmin, xmax]


def outlier_test(x):
    # flags outlier endpoints
    # x is an n x 2 vector of right and left interval endpoints
    # calculate box and whisker test on x[0] and x[1]
    x = np.array(x)
    x0 = box_and_whisker(x[:, 0])
    x1 = box_and_whisker(x[:, 1])
    y = []

    # eliminate outlier endpoints in x using box and whisker test
    for i in range(0, len(x)):
        if (x0[0] <= x[i, 0] <= x0[1] and x1[0] <= x[i, 1] <= x1[1]):
            y.append(x[i])
    if len(y) == 0:
        return None

    y = np.array(y)
    # y has first pass outliers eliminated
    # repeat box and whisker test on L=y[1]-y[0]
    L = y[:, 1] - y[:, 0]
    Lbw = box_and_whisker(L)
    z = []
    for i in range(0, len(y)):
        if (Lbw[0] <= L[i] <= Lbw[1]):
            z.append(y[i])
    if len(z) == 0:
        return None
    return np.array(z)


outlietest = [[0, 6],
              [2, 4],
              [3, 4],
              [5, 6],
              [4, 10],
              [5, 5.5],
              [5.2, 5.3]]


# print(outlier_test(outlietest))


def fr(y, a):
    r = y
    func = lambda r: norm.cdf(y + r) - norm.cdf(y - r) - (1 - a)
    return root(func, r).x[0]


def fk(xk, a, n):
    def integrand(y):
        term1 = ((n - 1) * fr(y, a) ** 2) / (xk ** 2)
        term2 = n - 1
        pchisq = 1 - chi2.cdf(term1, term2)
        term3 = (-1 / 2) * n * y ** 2
        return ((2 * np.sqrt(n)) / (np.sqrt(2 * np.pi))) * pchisq * np.exp(term3)

    result, error = quad(integrand, 0, np.inf)
    return result


def kk(a, y, n):
    xk = 2
    func = lambda xk: fk(xk, a, n) - (1 - y)
    return root(func, xk).x[0]


# print(kk(0.05,0.05,30))

def tolerance(x, m, sigma, k):
    # flags valuews of x where either endpoint is outside the tolerance limits
    upperlim = m + k * sigma
    lowerlim = m - k * sigma
    for i in range(0, len(x)):
        if lowerlim <= x[i] <= upperlim:
            continue
        else:
            x[i] = "Out of tolerance"
    return x


def reasonable(x, ml, mr, sigma_l, sigma_r):
    # eliminate unreasonable intervals that do not overlap all other intervals or have too little overlap
    # x is an n x 2 vector of intervals
    # ml, mr, sigma_l, sigma_r are means and standard deviations
    # If sigma_l or sigma_r is zero, all intervals overlap by definition
    if sigma_l == 0 or sigma_r == 0:
        return x
    # If sigma)l == sigma_r, the solution to Eq. (A-6) in Ref. [1] is the mean of the means
    if sigma_l == 0 or sigma_r == 0:
        return x
    if sigma_l == sigma_r:
        zeta = (ml + mr) / 2
    else:
        zeta_1 = ((mr * sigma_l ** 2 - ml * sigma_r ** 2) - sigma_l * sigma_r * np.sqrt(
            (ml - mr) ** 2 + 2 * (sigma_l ** 2 - sigma_r ** 2) * math.log(sigma_l / sigma_r))) / (
                             sigma_l ** 2 - sigma_r ** 2)
        zeta_2 = ((mr * sigma_l ** 2 - ml * sigma_r ** 2) + sigma_l * sigma_r * np.sqrt(
            (ml - mr) ** 2 + 2 * (sigma_l ** 2 - sigma_r ** 2) * math.log(sigma_l / sigma_r))) / (
                             sigma_l ** 2 - sigma_r ** 2)

        if ml <= zeta_1 <= mr:
            zeta = zeta_1
        elif ml <= zeta_2 <= mr:
            zeta = zeta_2
        else:
            zeta = zeta_1 if abs(ml - zeta_1) < abs(mr - zeta_2) else zeta_2

    y = []
    for interval in x:
        if (2 * ml - zeta <= interval[0] < zeta < interval[1] <= 2 * mr - zeta):
            y.append(interval)
    return y

# test = [[5, 6], [3, 4], [4, 7], [4, 6], [4, 7], [4, 5], [4.2, 5.3]]

# ml = mean([item[0] for item in test])
# mr = mean([item[1] for item in test])

# sigma_l = np.std([item[0] for item in test])
# sigma_r = np.std([item[1] for item in test])

# print(reasonable(test, ml, mr, sigma_l, sigma_r))
