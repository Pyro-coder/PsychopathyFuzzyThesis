import numpy as np
from scipy.optimize import brentq

def alpha_cut(mu, xmin, xmax, m, n):
    # Calculate the increment for the x values
    x_incr = (xmax - xmin) / n
    maxy = 0
    maxxleft = xmin
    maxxright = xmin

    # Initialize the array for alpha-cuts
    alphacuts = np.zeros((m + 1, 3))  # Each row: [alpha_left, alpha_right, alpha]

    # First, find the maximum value of the membership function to set the range
    for i in range(n + 1):
        x = xmin + i * x_incr
        mu_x = mu(x)
        if mu_x > maxy:
            maxy = mu_x
            maxxleft = maxxright = x
        elif mu_x == maxy:
            maxxright = x

    # Zero alpha cut interval
    alphacuts[0, :] = [xmin, xmax, 0]

    # Find non-zero alpha cut intervals
    for i in range(1, m + 1):
        alpha = i / m
        alphacuts[i, 2] = alpha  # Store alpha value

        if alpha > maxy:
            break  # No alpha-cuts above the maximum of the membership function

        # Define the function for finding roots
        def root_func(x):
            return mu(x) - alpha

        # Calculate left and right alpha-cuts
        if alpha < maxy:
            try:
                # Left alpha cut
                if mu(xmin) >= alpha:
                    alphacuts[i, 0] = xmin
                else:
                    alphacuts[i, 0] = brentq(root_func, xmin, maxxleft)

                # Right alpha cut
                if mu(xmax) >= alpha:
                    alphacuts[i, 1] = xmax
                else:
                    alphacuts[i, 1] = brentq(root_func, maxxright, xmax)
            except ValueError:
                # Handle cases where no root is found within the interval
                pass

    return alphacuts
