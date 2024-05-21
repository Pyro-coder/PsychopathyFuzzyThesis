import numpy as np
import matplotlib.pyplot as plt

from max_alpha_for_lmf import alpha_maxlmf
from EKM import t2wpm
from trapezoid import trap
from fou_points import fouset
from select_f import select_f
from type_1_fuzzy_weighted_average import alpha_fwa
from calculate_mf_from_alpha_cuts import mu_Sf
from t2_centroid import t2_centroid, defuzz


def alpha_to_alpha_t2wpm(aux, alx, auw, alw, r):
    """
    Calculate α-cuts of Type-2 fuzzy weighted power mean given UMF/LMF α-cuts of inputs.

    Parameters:
    aux, alx, auw, alw - Nested n-vectors, each element containing an array of α-cuts.
    r - Power parameter for the weighted power mean, which can be -∞, finite, or ∞.

    Returns:
    A nested 2-vector, each element containing an array of corresponding α-cut intervals of the WPM UMF/LMF.
    """
    n = len(aux)  # Number of terms in the weighted power mean
    m = len(aux[0])  # Number of α-cuts of UMF
    out = [None, None]

    # Process UMF
    for j in range(m):
        xuab = [aux[i][j] for i in range(n)]
        wucd = [auw[i][j] for i in range(n)]

        # Compute WPM α-cut for each UMF α-cut using an appropriate algorithm
        zu = t2wpm(xuab, wucd, r)  # Placeholder for the actual weighted power mean calculation
        if j == 0:
            out[0] = zu
        else:
            out[0] = np.vstack((out[0], zu))

    # Determine the minimum index and alpha value from the LMF α-cut arrays
    hmin, _ = alpha_maxlmf(alx, alw)

    # Process LMF for finite r
    if r != float('inf') and r != float('-inf'):
        for j in range(hmin + 1):
            xlab = [alx[i][j] for i in range(n)]
            wlcd = [alw[i][j] for i in range(n)]
            zl = t2wpm(xlab, wlcd, r)
            if j == 0:
                out[1] = zl
            else:
                out[1] = np.vstack((out[1], zl))

    # Handle r as -∞ or ∞
    elif r == float('-inf'):
        # Min interval logic for LMF
        xmin = [min(alx[i][j][0] for i in range(n)) for j in range(hmin + 1)]
        xmax = [min(alx[i][j][1] for i in range(n)) for j in range(hmin + 1)]
        out[1] = np.array([[xmin[j], xmax[j], j / m] for j in range(hmin + 1)])
    elif r == float('inf'):
        # Max interval logic for LMF
        xmin = [max(alx[i][j][0] for i in range(n)) for j in range(hmin + 1)]
        xmax = [max(alx[i][j][1] for i in range(n)) for j in range(hmin + 1)]
        out[1] = np.array([[xmin[j], xmax[j], j / m] for j in range(hmin + 1)])

    return out


def apply_trapezoidal_mfs(x, arrays):
    """Apply trapezoidal membership functions to x for a given list of parameter arrays."""
    results = []
    for array in arrays:
        # Calculate membership value for each array and add to results
        results.append(trap(x, *array))
    return np.array(results)


# Membership function parameters
zu0 = np.array([1, 0.1, 0.8, 1.2, 2])
zl0 = np.array([0.8, 0.5, 0.8, 1.2, 1.5])
wu0 = np.array([1, 0, 0.7, 1.3, 2])
wl0 = np.array([0.8, 0, 1, 1, 2])
zu1 = np.array([1, 0.1, 1, 2, 3])
zl1 = np.array([0.7, 0.5, 1.5, 1.8, 2.5])
wu1 = np.array([1, 0, 1.8, 2.2, 3])
wl1 = np.array([0.6, 1, 1.5, 2, 2.5])


def zu(x):
    """Compute trapezoidal MFs for zu parameters."""
    params = [zu0, zu1]  # zu1
    results = [trap(x, *param) for param in params]
    return np.array(results)


def zl(x):
    """Compute trapezoidal MFs for zl parameters."""
    params = [zl0, zl1]  # zl1
    results = [trap(x, *param) for param in params]
    return np.array(results)


def wu(x):
    """Compute trapezoidal MFs for wu parameters."""
    params = [wu0, wu1]  # wu1
    results = [trap(x, *param) for param in params]
    return np.array(results)


def wl(x):
    """Compute trapezoidal MFs for wl parameters."""
    params = [wl0, wl1]  # wl1
    results = [trap(x, *param) for param in params]
    return np.array(results)


def z0u(x):
    """Function to extract the first element from the vector returned by zu(x)."""
    return select_f(zu, x, 0)  # 0 indicates the first element


# Function to extract the second element from the vector returned by zu(x)
def z1u(x):
    return select_f(zu, x, 1)


# Function to extract the first element from the vector returned by zl(x)
def z0l(x):
    return select_f(zl, x, 0)


# Function to extract the second element from the vector returned by zl(x)
def z1l(x):
    return select_f(zl, x, 1)


# Function to extract the first element from the vector returned by wu(x)
def w0u(x):
    return select_f(wu, x, 0)


# Function to extract the second element from the vector returned by wu(x)
def w1u(x):
    return select_f(wu, x, 1)


# Function to extract the first element from the vector returned by wl(x)
def w0l(x):
    return select_f(wl, x, 0)


# Function to extract the second element from the vector returned by wl(x)
def w1l(x):
    return select_f(wl, x, 1)


z0fou = fouset(z0u, z0l, 0, 10, .01, 0.012)
z1fou = fouset(z1u, z1l, 0, 10, .01, 0.013)
w0fou = fouset(w0u, w0l, 0, 10, .01, 0.012)
w1fou = fouset(w1u, w1l, 0, 10, .01, 0.013)

# u = np.arange(0, 3.01, 0.01)

# jz0 = range(len(z0fou))
# jz1 = range(len(z1fou))


# x_coords_z0 = [coord[0] for coord in z0fou]
# y_coords_z0 = [coord[1] for coord in z0fou]

# x_coords_z1 = [coord[0] for coord in z1fou]
# y_coords_z1 = [coord[1] for coord in z1fou]

# plt.scatter(x_coords_z0, y_coords_z0, label='z0fou')
# plt.scatter(x_coords_z1, y_coords_z1, label='z1fou')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Scatter Plot of z0fou and z1fou')
# plt.legend()
# plt.show()


# Construct the support intervals matrix for zu and wu
xsup = np.array([
    [zu0[1], zu0[4]],  # Extracting the second and fifth elements of the first trapezoidal function
    [zu1[1], zu1[4]]  # Same extraction for the second function
])

wsup = np.array([
    [wu0[1], wu0[4]],  # Similarly for wu
    [wu1[1], wu1[4]]
])

# Call the αfwa function with the specified parameters
Au = alpha_fwa(zu, wu, xsup, wsup, 100, 300)

# Construct the support intervals matrix for zl and wl
xsup = np.array([
    [zl0[1], zl0[4]],  # Extracting specific elements for zl
    [zl1[1], zl1[4]]
])

wsup = np.array([
    [wl0[1], wl0[4]],  # Similarly for wl
    [wl1[1], wl1[4]]
])

# Call the αfwa function to compute the alpha cuts for the LMF
Al = alpha_fwa(zl, wl, xsup, wsup, 100, 300)

Al = Al[:61]


# TODO TOMORROW, figure out why the output of alpha_fwa is not 61 for al

def lwa_umf(x):
    return mu_Sf(x, Au)


def lwa_lmf(x):
    return mu_Sf(x, Al)


c = t2_centroid(Au, Al, 300)
# print(c)

lwa_FOU = fouset(lwa_umf, lwa_lmf, 0, 10, 0.05, 0.012)

cl = c[0]
cr = c[1]

m = defuzz(c)
# print(m)


def t2wpmtroid(Au, Al, N, r):
    """
    WPM centroid interval of Type-2 MF computed using WPM algorithm.
    Au and Al are arrays of α-cuts of upper and lower MFs; N is # of x slices.
    """
    xinc = (Au[0, 1] - Au[0, 0]) / N
    x = [Au[0, 0] + i * xinc for i in range(N + 1)]
    xx = np.column_stack((x, x))  # xx is an array of intervals of zero width

    # Generate lower and upper MF values for each x
    w = np.zeros((N + 1, 2))
    for i in range(N + 1):
        w[i, 0] = mu_Sf(x[i], Al)
        w[i, 1] = mu_Sf(x[i], Au)

    # Ensure w intervals are not zero width
    w[w[:, 1] - w[:, 0] == 0, 1] += 0.001

    # Compute centroid interval using WPMEKM
    centr = t2wpm(xx, w, r)
    out = np.array([centr[0], centr[1]])

    return out

# # 1. Create a range of x values
# x_values = np.arange(0, 10, 0.01)
#
# # 2. Calculate the corresponding y values for lwa_umf and lwa_lmf
# y_values_umf = [lwa_umf(x) for x in x_values]
# y_values_lmf = [lwa_lmf(x) for x in x_values]
#
# # 3. Plot lwa_umf and lwa_lmf
# plt.plot(x_values, y_values_umf, label='lwa_umf')
# plt.plot(x_values, y_values_lmf, label='lwa_lmf')
#
# # 4. Plot lwa_FOU
# x_coords_fou = [coord[0] for coord in lwa_FOU]
# y_coords_fou = [coord[1] for coord in lwa_FOU]
# plt.scatter(x_coords_fou, y_coords_fou, c='blue', s=1, label='lwa_FOU')
#
# # 5. Plot the centroid
# plt.scatter(c[0], 1, c='red', label='Centroid Left')
# plt.scatter(c[1], 1, c='red', label='Centroid Right')
#
# # 6. Plot the defuzzified centroid
# plt.scatter(m, 1, c='purple', label='Defuzzified Centroid')
#
# # 7. Add labels, a legend, and a title
# plt.xlabel('x')
# plt.ylabel('Membership Degree')
# plt.title('Plot of lwa_umf, lwa_lmf, lwa_FOU, Centroid, and Defuzzified Centroid')
# plt.legend()
#
# # 8. Display the plot
# plt.show()

print(Al)
print(t2wpmtroid(Au, Al, 1000, 0))

