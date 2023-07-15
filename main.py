import numpy as np
import scipy
from scipy.stats import nct
import scipy.optimize as opt

# here are the functions

# calculate the approxomation of the ratio of gamma(n+1)/gamma(n+3/2) for large m
def rgamma(n):
    x = n + 1
    fval = np.sqrt(x-.25) * (1 + 1 / (64 * x * x) + 1 / (128 * x * x * x))
    return 1/fval


# calculate the cdf of a non-central t statistic
# t is the cdf variable; n is # degrees of freedom; δ is the noncentrality parameter
def nct_cdf(t, n, lower_delta):
    return nct.cdf(t, n, lower_delta)





def inv_nct_cdf(P, n, delta):
    #Inverse of non-central t-statistic cdf; finds the threshold yielding a cdf value of P
    #n is the # of degrees of freedom

    # set initial t guess to lowercase delta (lower_delta) and step up and down to find an interval where f(t) has opposite signs for the endpoints
    def f(t):
        return nct.cdf(t, n, delta) - P

    # Initialize t to the quantile of the standard t-distribution
    t = scipy.stats.t.ppf(P, n)

    # Determine the direction to move in
    if f(t) < 0:
        # If the initial value is less than the target, move up
        t_upper = t
        while f(t_upper) < 0:
            t_upper *= 2
        t_lower = t_upper / 2
    else:
        # If the initial value is more than the target, move down
        t_lower = t
        while f(t_lower) > 0:
            t_lower /= 2
        t_upper = t_lower * 2

    # Once we have a bracket, use a root-finding algorithm to find the t value
    t_zero = opt.brentq(f, t_lower, t_upper)
    
    return t_zero

    
def ktol(upsilon, a, n):
    zp = scipy.stats.norm.ppf(1 - upsilon, 0, 1)
    lower_delta = zp * np.sqrt(n)
    result = inv_nct_cdf(1-a, n-1, lower_delta)/np.sqrt(n)
    return result

print (ktol(.05, .05, 175))