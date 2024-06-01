from matplotlib import pyplot as plt

import EKM
import trapezoid as tpz
import t2_centroid as t2c
import numpy as np
import alpha_cut_power_mean as acpm
import scipy.optimize as opt
import scipy.integrate as integrate
import mystic.symbolic as ms
from mystic.solvers import fmin
from mystic.penalty import quadratic_inequality
from mystic.constraints import as_constraint


# Define the gcd function
def gcd(x, w, r):
    if r <= 0:
        return EKM.t2wpm(x, w, r)
    else:
        xx = np.zeros_like(x)
        for i in range(len(x)):
            xx[i, 0] = 1 - x[i, 1]
            xx[i, 1] = 1 - x[i, 0]
        out = EKM.t2wpm(xx, w, -r)
        return np.array([1 - out[1], 1 - out[0]])


# Define the zcd function for CD variant
def zcd(x, y, w1, w2, r1, r2):
    if r1 == 0:
        out = ((1 - w1) * (x ** (r2 * w2)) * (y ** (r2 * (1 - w2))) + w1 * (x ** r2)) ** (1 / r2)
    else:
        out = ((1 - w1) * ((w2 * (x ** r1) + (1 - w2) * (y ** r1)) ** (r2 / r1)) + w1 * (x ** r2)) ** (1 / r2)

    return out


# Define the zdc function for DC variant
def zdc(x, y, w1, w2, r1, r2):
    if r2 == 0:
        out = ((w1 * (x ** r1) + (1 - w1) * (y ** r1)) ** ((1 - w2) / r1)) * (x ** w2)
    else:
        out = ((1 - w2) * ((w1 * (x ** r1) + (1 - w1) * (y ** r1)) ** (r2 / r1)) + w2 * (x ** r2)) ** (1 / r2)
    return out


# Define the andness function
def andness(r):
    def integrand_r0(x, y):
        return np.sqrt(x * y)

    def integrand_r(x, y, r):
        return ((x ** r + y ** r) / 2) ** (1 / r)

    if r == 0:
        integral, _ = integrate.dblquad(integrand_r0, 0, 1, lambda y: 0, lambda y: 1, epsabs=1e-10, epsrel=1e-10)
        out = 2 - 3 * integral
    else:
        try:
            integral, _ = integrate.dblquad(integrand_r, 0, 1, lambda y: 0, lambda y: 1, args=(r,), epsabs=1e-10,
                                            epsrel=1e-10)
            out = 2 - 3 * integral
        except Exception:
            if r > 1:
                out = 0
            else:
                out = 1
    return out


# Define the omega function
def omega(r):
    return 1 - andness(r)


# Define the r_exponent function
def r_exponent(alpha):
    if alpha == 1:
        return -np.inf
    elif alpha == 0:
        return np.inf
    elif 0 < alpha < 1:
        f = lambda r: andness(r) - alpha
        sol = opt.root(f, 0)
        return sol.x[0]
    else:
        raise ValueError("Alpha should be between 0 and 1")


# Define the dc_delta_plus function for DC variant
def dc_delta_plus(w1, w2, r1, r2):
    def integrand(x):
        return zdc(x, 1, w1, w2, r1, r2)

    integral_value, _ = integrate.quad(integrand, 0, 1, epsabs=1e-10, epsrel=1e-10)
    delta_plus = 100 * abs(2 * integral_value - 1)

    return delta_plus


# Define the dc_delta_minus function for DC variant
def dc_delta_minus(w1, w2, r1, r2):
    if r2 != 0:
        out = 100 * (((1 - w2) * (w1 ** (r2 / r1)) + w2) ** (1 / r2) - 1)
    else:
        out = 0
    return out


# Define the cd_delta_plus function for CD variant
def cd_delta_plus(w1, w2, r1, r2):
    """
    Calculate δ+ for CD variant
    Note that r1 < 1, r2 >= 1 in this variant, and a = w1, b = w2 in Dujmovic's paper
    δ+ is the specified (Reward) truth value in % of a CD variant partial absorption operator when the absorption controlled variable y = 0
    δ+ and δ- are used to determine the weights w1 and w2 of partial absorption operator
    """
    integrand = lambda x: zcd(x, 1, w1, w2, r1, r2)

    integral_result, _ = integrate.quad(integrand, 0, 1)

    delta_plus = 100 * abs(2 * integral_result - 1)

    return delta_plus


# Define the cd_delta_minus function for CD variant
def cd_delta_minus(w1, w2, r1, r2):
    if r2 != 0:
        if r1 > 0:
            out = 100 * ((1 - w1) * (w2 ** (r2 / r1)) + w1) ** (1 / r2) - 100
        else:
            out = 100 * (w1 ** (1 / r2)) - 100
    else:
        out = 0

    return out


# Define the objective function for DC variant
def objective_dc(params, r1, r2, reward, penalty):
    w1, w2 = params
    delta_plus = dc_delta_plus(w1, w2, r1, r2)
    delta_minus = dc_delta_minus(w1, w2, r1, r2)
    return (delta_plus - reward) ** 2 + (delta_minus - penalty) ** 2


# Define the objective function for CD variant
def objective_cd(params, r1, r2, reward, penalty):
    w1, w2 = params
    delta_plus = cd_delta_plus(w1, w2, r1, r2)
    delta_minus = cd_delta_minus(w1, w2, r1, r2)
    return (delta_plus - reward) ** 2 + (delta_minus - penalty) ** 2


# Define the constraints
def constraints(params):
    w1, w2 = params
    return [w1, 1 - w1, w2, 1 - w2]


# Set up penalty functions for the constraints
@quadratic_inequality(lambda w1_w2: w1_w2[0] >= 0)
@quadratic_inequality(lambda w1_w2: w1_w2[0] <= 1)
@quadratic_inequality(lambda w1_w2: w1_w2[1] >= 0)
@quadratic_inequality(lambda w1_w2: w1_w2[1] <= 1)
def penalty(params):
    return 0.0


# Define the wcd function for DC variant
def wdc(r1, r2, P, R):
    # Set the initial guess
    initial_guess = [0.5, 0.5]

    # Optimize
    result = fmin(objective_dc, x0=initial_guess, penalty=penalty, args=(r1, r2, R, P))

    # Extract the optimized values of w1 and w2
    w1_opt, w2_opt = result

    return np.array([w1_opt, w2_opt])


# Define the wcd function for CD variant
def wcd(r1, r2, P, R):
    # Set the initial guess
    initial_guess = [0.5, 0.5]

    # Optimize
    result = fmin(objective_cd, x0=initial_guess, penalty=penalty, args=(r1, r2, R, P))

    # Extract the optimized values of w1 and w2
    w1_opt, w2_opt = result

    return np.array([w1_opt, w2_opt])


wAR = wdc(1, r_exponent(14 / 16), -25, 15)


# print(wAR)
#
# print(dc_delta_plus(wAR[0], wAR[1], 1, r_exponent(14 / 16)))
# print(dc_delta_minus(wAR[0], wAR[1], 1, r_exponent(14 / 16)))
# r2 = r_exponent(14 / 16)


def fAR2(x, y):
    return zdc(x, y, wAR[0], wAR[1], 1, r2)


# # Create a mesh grid
# x = np.linspace(0.001, 1, 50)
# y = np.linspace(0.001, 1, 50)
# X, Y = np.meshgrid(x, y)
#
# # Evaluate the function over the mesh grid using the safe wrapper
# Z = np.vectorize(fAR2)(X, Y)
#
# # Ensure Z contains finite values for contour plot
# Z = np.nan_to_num(Z, nan=np.nanmin(Z))
#
# # Define contour levels at 0.1 intervals
# levels = np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1)
#
# # Plot the mesh grid
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
# cbar = plt.colorbar(contour, ticks=np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1))
# cbar.ax.set_ylabel('fAR2(x, y)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour plot of fAR2(x, y)')
# plt.show()

wRA = wcd(r_exponent(14 / 16), 1, -25, 15)


# print(wRA)
#
# print(cd_delta_plus(wRA[0], wRA[1], r_exponent(14 / 16), 1))
# print(cd_delta_minus(wRA[0], wRA[1], r_exponent(14 / 16), 1))
#
# r1 = r_exponent(14 / 16)
# print(r1)

def fRA2(x, y):
    return zcd(x, y, wRA[0], wRA[1], r1, 1)


# # Create a mesh grid
# x = np.linspace(0.001, 1, 50)
# y = np.linspace(0.001, 1, 50)
# X, Y = np.meshgrid(x, y)
#
# # Evaluate the function over the mesh grid using the safe wrapper
# Z = np.vectorize(fRA2)(X, Y)
#
# # Ensure Z contains finite values for contour plot
# Z = np.nan_to_num(Z, nan=np.nanmin(Z))
#
# # Define contour levels at 0.1 intervals
# levels = np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1)
#
# # Plot the mesh grid
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
# cbar = plt.colorbar(contour, ticks=np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1))
# cbar.ax.set_ylabel('fRA2(x, y)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour plot of fRA2(x, y)')
# plt.show()

wRRbar = wcd(0, r_exponent(7 / 16), -25, 15)
r2 = r_exponent(7 / 16)


# print(cd_delta_plus(wrp_bar[0], wrp_bar[1], 0, r_exponent(7 / 16)))
# print(cd_delta_minus(wrp_bar[0], wrp_bar[1], 0, r_exponent(7 / 16)))
# print(r2)


def fRRbar2(x, y):
    return zcd(x, y, wRRbar[0], wRRbar[1], 0, r2)


# # Create a mesh grid
# x = np.linspace(0.001, 1, 50)
# y = np.linspace(0.001, 1, 50)
# X, Y = np.meshgrid(x, y)
#
# # Evaluate the function over the mesh grid using the safe wrapper
# Z = np.vectorize(fRRbar2)(X, Y)
#
# # Ensure Z contains finite values for contour plot
# Z = np.nan_to_num(Z, nan=np.nanmin(Z))
#
# # Define contour levels at 0.1 intervals
# levels = np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1)
#
# # Plot the mesh grid
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
# cbar = plt.colorbar(contour, ticks=np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1))
# cbar.ax.set_ylabel('fRRbar2(x, y)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour plot of fRRbar2(x, y)')
# plt.show()

wRbarR = wdc(r_exponent(7 / 16), -10, -25, 15)
r1 = r_exponent(7 / 16)


def fRbarR2(x, y):
    return zdc(x, y, wRbarR[0], wRbarR[1], r1, -10)


# # Create a mesh grid
# x = np.linspace(0.001, 1, 50)
# y = np.linspace(0.001, 1, 50)
# X, Y = np.meshgrid(x, y)
#
# # Evaluate the function over the mesh grid using the safe wrapper
# Z = np.vectorize(fRbarR2)(X, Y)
#
# # Ensure Z contains finite values for contour plot
# Z = np.nan_to_num(Z, nan=np.nanmin(Z))
#
# # Define contour levels at 0.1 intervals
# levels = np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1)
#
# # Plot the mesh grid
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
# cbar = plt.colorbar(contour, ticks=np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1))
# cbar.ax.set_ylabel('fRbarR2(x, y)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour plot of fRbarR2(x, y)')
# plt.show()


def one_minus(x):
    """
    Compute (1 - x) α-cuts of an array of α-cuts x.
    """
    rows = x.shape[0]
    out = np.zeros_like(x)

    for i in range(rows):
        out[i, 0] = 1 - x[i, 1]
        out[i, 1] = 1 - x[i, 0]
        out[i, 2] = x[i, 2]

    return out


def alpha_w(w, n):
    """
    Compute (n + 1) α-cuts array and its complement for a constant w.
    """
    wout = np.full((n + 1,), w)
    alpha = np.linspace(0, 1, n + 1)

    # Create the augmented array
    augmented = np.column_stack((wout, wout, alpha))

    # Compute the complement
    complement = one_minus(augmented)

    # Combine the results into a single output array
    out = [augmented, complement]

    return out


def alpha_cpa(x, y, r1, r2, P, R):
    """
    Compute α-cuts of DC-variant conjunctive partial absorption operator,
    with x mandatory, y desired IT2 fuzzy variables.
    """
    # Compute w1 and w2 corresponding to P and R
    w = wdc(r1, r2, P, R)

    # Calculate DCδplus and DCδminus to verify correct weights
    dcδplus = dc_delta_plus(w[0], w[1], r1, r2)
    dcδminus = dc_delta_minus(w[0], w[1], r1, r2)

    # Compute the α-cuts of the disjunction of x and y, using weight w_0, exponent r1 >= 1
    disj = acpm.alpha_to_alpha_t2wpm([[x[0], y[0]], [x[1], y[1]]], [[x[0], y[0]], [x[1], y[1]]], alpha_w(w[0], 100)[0],
                                     alpha_w(w[0], 100)[1], r1)

    # Compute the α-cuts of the conjunction of x and disj, using weight w_1, exponent r2 < 1
    out = acpm.alpha_to_alpha_t2wpm([[x[0], disj[0]], [x[1], disj[1]]], [[x[0], disj[0]], [x[1], disj[1]]],
                                    alpha_w(w[1], 100)[0], alpha_w(w[1], 100)[1], r2)

    return out


def alpha_dpa(x, y, r1, r2, P, R):
    """
    Compute α-cuts of CD-variant disjunctive partial absorption operator,
    with x sufficient, y desired IT2 fuzzy variables.
    """
    # Compute w1 and w2 corresponding to P and R
    w = wcd(r1, r2, P, R)

    # Calculate CDδplus and CDδminus to verify correct weights
    cdδplus = cd_delta_plus(w[0], w[1], r1, r2)
    cdδminus = cd_delta_minus(w[0], w[1], r1, r2)

    # Compute the α-cuts of the conjunction of x and y, using weight w_0, exponent r1 < 1
    conj = acpm.alpha_to_alpha_t2wpm([[x[0], y[0]], [x[1], y[1]]], [[x[0], y[0]], [x[1], y[1]]], alpha_w(w[0], 100)[0],
                                     alpha_w(w[0], 100)[1], r1)

    # Compute the α-cuts of the disjunction of x and conj, using weight w_1, exponent r2 >= 1
    out = acpm.alpha_to_alpha_t2wpm([[x[0], conj[0]], [x[1], conj[1]]], [[x[0], conj[0]], [x[1], conj[1]]],
                                    alpha_w(w[1], 100)[0], alpha_w(w[1], 100)[1], r2)

    return out


x = np.arange(0, 10.01, 0.01)


def NVLl(x):
    return tpz.trap(x, 1, 0, 0, 0.02, 0.33)


def VLl(x):
    return tpz.trap(x, 1, 0, 0, 0.14, 1.82)


def Ll(x):
    return tpz.trap(x, 0.31, 1.9, 2.24, 2.24, 2.51)


def MLLl(x):
    return tpz.trap(x, 0.32, 2.99, 3.31, 3.31, 3.81)


def FMLHl(x):
    return tpz.trap(x, 0.43, 5.79, 6.31, 6.31, 7.21)


def MLHl(x):
    return tpz.trap(x, 0.29, 6.9, 7.21, 7.21, 7.6)


def Hl(x):
    return tpz.trap(x, 1, 7.68, 9.82, 9.82, 10)


def EHl(x):
    return tpz.trap(x, 1, 9.74, 9.98, 9.98, 10)


def NVLu(x):
    return tpz.trap(x, 1, 0, 0, 0.22, 3.16)


def VLu(x):
    return tpz.trap(x, 1, 0, 0, 1.37, 3.95)


def Lu(x):
    return tpz.trap(x, 1, 0.38, 1.63, 3, 4.62)


def MLLu(x):
    return tpz.trap(x, 1, 0.38, 2.25, 4, 5.92)


def FMLHu(x):
    return tpz.trap(x, 1, 2.33, 5.11, 7, 9.59)


def MLHu(x):
    return tpz.trap(x, 1, 4.38, 6.25, 8, 9.62)


def Hu(x):
    return tpz.trap(x, 1, 4.73, 8.82, 10, 10)


def EHu(x):
    return tpz.trap(x, 1, 7.1, 9.8, 10, 10)


def VBl(x):
    return tpz.trap(x, 1, 0, 0, 0.09, 1.32)


def Bl(x):
    return tpz.trap(x, 0.48, 1.79, 2.37, 2.37, 2.71)


def SBl(x):
    return tpz.trap(x, 0.42, 2.79, 3.3, 3.3, 3.71)


def Fl(x):
    return tpz.trap(x, 0.27, 4.79, 5.12, 5.12, 5.35)


def SGl(x):
    return tpz.trap(x, 0.4, 5.89, 6.34, 6.34, 6.81)


def Gl(x):
    return tpz.trap(x, 0.47, 6.79, 7.25, 7.25, 7.91)


def VGl(x):
    return tpz.trap(x, 1, 7.66, 9.82, 9.82, 10)


def VBu(x):
    return tpz.trap(x, 1, 0, 0, 0.59, 3.95)


def Bu(x):
    return tpz.trap(x, 1, 0.28, 2, 3, 5.22)


def SBu(x):
    return tpz.trap(x, 1, 0.98, 2.75, 4, 5.41)


def Fu(x):
    return tpz.trap(x, 1, 2.38, 4.5, 6, 8.18)


def SGu(x):
    return tpz.trap(x, 1, 4.02, 5.65, 7, 8.41)


def Gu(x):
    return tpz.trap(x, 1, 4.38, 6.5, 7.75, 9.62)


def VGu(x):
    return tpz.trap(x, 1, 5.21, 8.27, 10, 10)


def Ul(x):
    return tpz.trap(x, 1, 0, 0, 0.09, 1.15)


def MLUl(x):
    return tpz.trap(x, 0.34, 2.79, 3.21, 3.21, 3.71)


def MLIl(x):
    return tpz.trap(x, 0.33, 5.79, 6.28, 6.28, 6.67)


def VIl(x):
    return tpz.trap(x, 1, 8.68, 9.91, 9.91, 10)


def Uu(x):
    return tpz.trap(x, 1, 0, 0, 0.55, 4.61)


def MLUu(x):
    return tpz.trap(x, 1, 0.42, 2.25, 4, 5.41)


def MLIu(x):
    return tpz.trap(x, 1, 3.38, 5.5, 7.25, 9.02)


def VIu(x):
    return tpz.trap(x, 1, 7.37, 9.36, 10, 10)


def antonym(f, x):
    return f(10 - x)


def complement(f, x):
    return 1 - f(x)


def z8l(x):
    return np.array([
        NVLl(x),
        VLl(x),
        Ll(x),
        MLLl(x),
        FMLHl(x),
        MLHl(x),
        Hl(x),
        EHl(x)
    ])


def z8u(x):
    return np.array([
        NVLu(x),
        VLu(x),
        Lu(x),
        MLLu(x),
        FMLHu(x),
        MLHu(x),
        Hu(x),
        EHu(x)
    ])


def z7l(x):
    return np.array([
        VBl(x),
        Bl(x),
        SBl(x),
        Fl(x),
        SGl(x),
        Gl(x),
        VGl(x)
    ])


def z7u(x):
    return np.array([
        VBu(x),
        Bu(x),
        SBu(x),
        Fu(x),
        SGu(x),
        Gu(x),
        VGu(x)
    ])


def z4l(x):
    return np.array([
        Ul(x),
        MLUl(x),
        MLIl(x),
        VIl(x)
    ])


def z4u(x):
    return np.array([
        Uu(x),
        MLUu(x),
        MLIu(x),
        VIu(x)
    ])
