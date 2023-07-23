from statistics import mean

import numpy as np
import openpyxl

from noncenteral_tstatic_cdf import bad_data, outlier_test, kk, tolerance


def dataclean(x, x0, x1, a, significance_level):
    # preprocess raw interval data for a given word to eliminate unacceptable intervals using all of the above tests
    # x is an n x 2 array of interval endpoints; x0, x1 are bound on valid endpoints (typically 0 and 10 or 0 and 1)
    # stage 1 --Bad data elimination
    y = bad_data(x, x0, x1)
    if y is None:
        return "All intervals eliminated at bad data stage"

    # stage 2 -- Outlier elimination
    z = outlier_test(y)
    if z is None:
        return "All intervals eliminated at outlier elimination stage"
    mleft = np.mean(z[:, 0])
    sleft = np.std(z[:, 0])
    mright = np.mean(z[:, 1])
    sright = np.std(z[:, 1])

    # stage 3 -- Tolerance limit processing
    # k is the tolerance factor (multiple of stddev) wrt the interval endpoint means
    k = kk(a, significance_level, len(z))
    y0 = tolerance(z[:, 0], mleft, sleft, k)
    y1 = tolerance(z[:, 1], mright, sright, k)
    z = [[y0[i], y1[i]] for i in range(len(y0)) if y0[i] != "Out of tolerance" and y1[i] != "Out of tolerance"]

    # Perform tolerance test on the surviving interval lengths
    L = np.array(z)[:, 1] - np.array(z)[:, 0]
    k1 = kk(a, significance_level, len(L))
    k2 = np.mean(L) / np.std(L)
    k3 = (10 - np.mean(L)) / np.std(L)
    kprime = min(k1, k2, k3)
    Lt = tolerance(L, np.mean(L), np.std(L), kprime)
    zz = [[z[i][0], z[i][1]] for i in range(len(z)) if Lt[i] != "Out of tolerance"]
    if len(zz) == 0:
        return "All intervals eliminated at tolerance limit processing stage"

    # stage 4 -- Reasonable interval processing
    z = [[y0[i], y1[i]] for i in range(len(y0)) if y0[i] != "Out of tolerance" and y1[i] != "Out of tolerance"]

    if len(z) == 0:
        return "All intervals eliminated at reasonable interval processing stage"

    # compute sample means of residual interval endpoints
    out = [z, np.mean(np.array(z)[:, 0]), np.mean(np.array(z)[:, 1])]
    return out



ws = openpyxl.load_workbook('excel\Very Bad interval data.xlsx')["Sheet1"]
xVeryBad = []
for row in ws.iter_rows(min_row=1, max_row=40, min_col=1, max_col=2, values_only=True):
    xVeryBad.append([float(cell) for cell in row if cell is not None])

print(xVeryBad)

yVB = dataclean(xVeryBad, 0, 10, 0.05, 0.05)[0]

print(yVB)

xBad = openpyxl.load_workbook('excel\Bad interval data.xlsx')['Sheet1']
xSomewhatBad = openpyxl.load_workbook('excel\Somewhat Bad interval data.xlsx')['Sheet1']
xFair = openpyxl.load_workbook('excel\Fair interval data.xlsx')['Sheet1']
xSomewhatGood = openpyxl.load_workbook('excel\Somewhat Good interval data.xlsx')['Sheet1']
xGood = openpyxl.load_workbook('excel\Good interval data.xlsx')['Sheet1']
xVeryGood = openpyxl.load_workbook('excel\Very Good interval data.xlsx')['Sheet1']
