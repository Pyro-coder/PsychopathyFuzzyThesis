import numpy as np
import scipy
from scipy.stats import nct

# here are the functions

# calculate the approxomation of the ratio of gamma(n+1)/gamma(n+3/2) for large m
def rgamma(n):
    x = n + 1
    fval = np.sqrt(x-.25) * (1 + 1 / (64 * x * x) + 1 / (128 * x * x * x))
    return 1/fval


# calculate the cdf of a non-central t statistic
# t is the cdf variable; n is # degrees of freedom; δ is the noncentrality parameter, tol is the sum stopping tolerance
def nctcdf(t, n, lower_delta):
    return nct.cdf(t, n, lower_delta)


print (nctcdf(2, 100, 2))