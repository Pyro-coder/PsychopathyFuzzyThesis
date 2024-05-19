import numpy as np


def mu_Sf(x, A):
    """
    Type-1 MF corresponding to an (m+1)x3 array of α-cuts A.
    """
    mu = 0  # Initialize membership value to 0
    m = A.shape[0]  # Number of alpha-cuts

    # Iterate over each row in the alpha-cuts array
    for j in range(m):
        # Check if x lies within the current alpha-cut interval
        if A[j, 0] <= x <= A[j, 1]:
            mu = A[j, 2]  # Update mu to the alpha value of the current interval

    # Check for zero-length interval at the last alpha-cut
    if A[m - 1, 0] == A[m - 1, 1] and abs(x - A[m - 1, 0]) < 0.001:
        mu = A[m - 1, 2]  # Set mu to the alpha value if x is close enough to the point

    return mu