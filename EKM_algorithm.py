import numpy as np

def wa(x, w):
    """ Compute the weighted average of x with weights w. """
    num = np.dot(w.T, x)  # Dot product of weights and values
    den = np.sum(w)       # Sum of weights
    return num / den


def selectf(z, x, i):
    """ Select the ith component of the vector function z(x). """
    return z(x)[i]


def lIWA(x, c, d, k):
    """ Compute left interval bound. """
    num1 = np.sum(x[:k] * d[:k])
    num2 = np.sum(x[k:] * c[k:])
    den1 = np.sum(d[:k])
    den2 = np.sum(c[k:])
    return (num1 + num2) / (den1 + den2)

def rIWA(x, c, d, k):
    """ Compute right interval bound. """
    num1 = np.sum(x[:k] * c[:k])
    num2 = np.sum(x[k:] * d[k:])
    den1 = np.sum(c[:k])
    den2 = np.sum(d[k:])
    return (num1 + num2) / (den1 + den2)


import numpy as np

def kmalg(x, w):
    N = len(x)
    
    # Processing for yL
    aug = sorted(zip(x[:, 0], w[:, 0], w[:, 1]), key=lambda a: a[0])
    xminsort, wminsort, wmaxsort = zip(*aug)

    k = round(N / 2.4)
    k = max(k, 1)

    a = sum(xminsort[i] * wmaxsort[i] for i in range(k)) + sum(xminsort[i] * wminsort[i] for i in range(k, N))
    b = sum(wmaxsort[i] for i in range(k)) + sum(wminsort[i] for i in range(k, N))

    y = a / b
    kprime = next((i for i in range(1, N) if xminsort[i - 1] <= y <= xminsort[i]), None)

    while kprime and kprime != k:
        s = np.sign(kprime - k)
        a_prime = a + s * sum((xminsort[i] * (wmaxsort[i] - wminsort[i]) for i in range(min(k, kprime), max(k, kprime))))
        b_prime = b + s * sum((wmaxsort[i] - wminsort[i] for i in range(min(k, kprime), max(k, kprime))))
        y_prime = a_prime / b_prime

        y = y_prime
        a = a_prime
        b = b_prime
        k = kprime

    yL = y
    kL = k - 1

    # Processing for yR
    aug = sorted(zip(x[:, 1], w[:, 0], w[:, 1]), key=lambda a: a[0])
    xmaxsort, wminsort, wmaxsort = zip(*aug)

    k = round(N / 1.7)
    k = max(k, 1)

    a = sum(xmaxsort[i] * wminsort[i] for i in range(k)) + sum(xmaxsort[i] * wmaxsort[i] for i in range(k, N))
    b = sum(wminsort[i] for i in range(k)) + sum(wmaxsort[i] for i in range(k, N))

    y = a / b
    kprime = next((i for i in range(1, N) if xmaxsort[i - 1] <= y <= xmaxsort[i]), None)

    while kprime and kprime != k:
        s = np.sign(kprime - k)
        a_prime = a - s * sum((xmaxsort[i] * (wmaxsort[i] - wminsort[i]) for i in range(min(k, kprime), max(k, kprime))))
        b_prime = b - s * sum((wmaxsort[i] - wminsort[i] for i in range(min(k, kprime), max(k, kprime))))
        y_prime = a_prime / b_prime

        y = y_prime
        a = a_prime
        b = b_prime
        k = kprime

    yR = y
    kR = k - 1

    return yL, yR, kL, kR


# Example data input
x = np.array([[0.5, 1.5], [1.8, 4.2], [3.7, 6.3], [5.5, 8.5], [8.2, 9.8]])
w = np.array([[1, 3], [0.6, 1.4], [5.1, 6.9], [2.4, 5.6], [7, 9]])

# Calculate the Karnik-Mendel algorithm result
numberresult = kmalg(x, w)
print("Result of kmalg:", numberresult)

print((numberresult[0] + numberresult[1])/2)