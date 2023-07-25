import noncenteral_tstatic_cdf
import openpyxl
import numpy as np
import nct_cdf
import dataclean


def hma_fou_class0(x, x0, x1):
    # Step 1 of HMA: determine the FOU class for naturally bounded (class 0) sets
    # x is an n x 2 array of intervals that survived the dataclean process
    # [x0, x1] is the bound interval
    ml = np.mean(x[:, 0])
    sigma_l = np.std(x[:, 0])
    mr = np.mean(x[:, 1])
    sigma_r = np.std(x[:, 1])
    k = nct_cdf.ktol(0.05, 0.05, len(x), 10 ** -5)
    al = ml - k * sigma_l
    bu = mr + k * sigma_r
    if al <= x0:
        out = "Left shoulder FOU"
    else:
        if bu >= x1:
            out = "Right shoulder FOU"
        else:
            out = "Interior FOU"
    return out

def hma_fou_class1(x, x0):
    # Step 1 of HMA: determine the FOU class for sets bounded only on the left (class 1)
    # In these cases, there are only Left shoulder or Interior FOUs
    # x is an n x 2 array of intervals that survived the dataclean processing
    # x0 is the left bound (typically 0) for the intervals
    ml = np.mean(x[:, 0])
    sigma_l = np.std(x[:, 0])
    k = nct_cdf.ktol(0.05, 0.05, len(x), 10 ** -5)
    al = ml - k * sigma_l
    if al <= x0:
        out = "Left shoulder FOU"
    else:
        out = "Interior FOU"
    return out


# debugging tests
# print(hma_fou_class0(dataclean.yVB, 0, 10))
# print(hma_fou_class0(dataclean.yB, 0, 10))
# print(hma_fou_class0(dataclean.ySB, 0, 10))
# print(hma_fou_class0(dataclean.yF, 0, 10))
# print(hma_fou_class0(dataclean.ySG, 0, 10))
# print(hma_fou_class0(dataclean.yG, 0, 10))
# print(hma_fou_class0(dataclean.yVG, 0, 10))

def hma_overlap(x, c, x0, x1):
    # Compute overlap interval of x rows for FOU class c
    # Note: the data part eliminates non-overlapping intervals, so we are assured of overlap
    if c == "Left shoulder FOU":
        out = [x0, min(x[:, 1])]
    elif c == "Interior FOU":
        out = [max(x[:, 0]), min(x[:, 1])]
    elif c == "Right shoulder FOU":
        out = [max(x[:, 0]), x1]
    return out

# tests for debugging hma_overlap
# print(hma_overlap(dataclean.yVB, "Left shoulder FOU", 0, 10))
# print(hma_overlap(dataclean.yB, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.ySB, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.yF, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.ySG, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.yG, "Interior FOU", 0, 10))
# print(hma_overlap(dataclean.yVG, "Right shoulder FOU", 0, 10))

def hma_olap_remove(x, c):
    # Given interval array x and FOU class, remove overlap from original intervals
    # For Left or Right shoulder FOUs, this leaves a single array of smaller intervals
    # For Interior FOUs, this leaves a nested 2-vector of arrays of smaller intervals for the left/right sections of the FOU, respectively
    out = []
    if c == "Left shoulder FOU":
        bmin = min(x[:, 1])
        for i in range(len(x)):
            out.append([bmin, x[i, 1]])
    elif c == "Right shoulder FOU":
        amax = max(x[:, 0])
        for i in range(len(x)):
            out.append([x[i, 0], amax])
    elif c == "Interior FOU":
        out0 = []
        out1 = []
        amax = max(x[:, 0])
        bmin = min(x[:, 1])
        for i in range(len(x)):
            out0.append([x[i, 0], amax])
            out1.append([bmin, x[i, 1]])
        out = [out0, out1]
    return out

# Debugging tests for hma_olap_remove
# VBr = hma_olap_remove(dataclean.yVB, "Left shoulder FOU")
# print(VBr)

# Br = hma_olap_remove(dataclean.yB, "Interior FOU")
# print(Br)

# SGr = hma_olap_remove(dataclean.ySG, "Interior FOU")
# print(SGr[0])
# print(SGr[1])

# Gr = hma_olap_remove(dataclean.yG, "Interior FOU")
# print(Gr)

VGr = hma_olap_remove(dataclean.yVG, "Right shoulder FOU")
# print(VGr)

def aleft(xr, x0):
    # Calculate a parameters for left-hand side of Interior or Right shoulder FOU
    # See eq. (5) in HMA paper
    # xr is the set of reduced intervals for the left-hand side from the hmaolapremove function
    # x0 is the left bound
    # oleft is the left bound of the overlap interval of the original interval set
    # Thus, oleft will be the max right bound of the reduced intervals xr
    oleft = max(xr[:, 0])
    intlengths = np.abs(xr[:, 0] - oleft)
    # mLH is the mean of interval lengths wrt the left bound of the overlap interval (oleft)
    mLH = oleft - np.mean(intlengths)
    sLH = np.std(intlengths)
    aleft = max(x0, oleft - 3 * np.sqrt(2) * sLH)
    aright = min(oleft, 6 * mLH + 3 * np.sqrt(2) * sLH - 5 * oleft)
    # Now test to if the order is sensible, and if not, reverse them
    if aleft <= aright:
        out = [aleft, aright]
    else:
        out = [max(x0, aright), min(oleft, aleft)]
    # out_0/out_1 are the UMF/LMF intersections with the x-axis, respectively
    return out

def aright(xr, xl):
    # Calculate a parameters for right-hand side of Interior or Left shoulder FOU
    # See eq. (6) in HMA paper
    # xr is the set of reduced intervals for the right-hand side from the hmaolapremove function
    # x1 is the right bound
    # oright is the right bound of the overlap interval of the original interval set
    # Thus, oright will be the min left bound of the reduced intervals xr
    oright = min(xr[:, 1])
    intlengths = np.abs(xr[:, 1] - oright)
    #mRH is the mean of interval length wrt the right bound of the overlap interval (oright)
    mRH = oright + np.mean(intlengths)
    sRH = np.std(intlengths)
    bright = min(xl, oright + 3 * np.sqrt(2) * sRH)
    bleft = max(oright, 6 * mRH - 3 * np.sqrt(2) * sRH - 5 * xl)
    # Now test to if the order is sensible, and if not, reverse them
    if bleft <= bright:
        out = [bleft, bright]
    else:
        out = [max(oright, bright), min(xl, bleft)]
    # out_0/out_1 are the UMF/LMF intersections with the x-axis, respectively
    return out

# Debugging tests for aleft and aright
print(aleft(VGr, 0))