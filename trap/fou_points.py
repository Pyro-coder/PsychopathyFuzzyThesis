import numpy as np

def umf(x, ml, mr, sigma, ks):
    if ml - ks * sigma <= x <= ml:
        return np.exp(-0.5 * ((x - ml) / sigma) ** 2)
    elif ml <= x <= mr:
        return 1
    elif mr < x <= mr + ks * sigma:
        return np.exp(-0.5 * ((x - mr) / sigma) ** 2)
    else:
        return 0

def lmf(x, ml, mr, sigma, ks):
    if mr - ks * sigma <= x <= (ml + mr) / 2:
        return np.exp(-0.5 * ((x - mr) / sigma) ** 2)
    elif (ml + mr) / 2 < x <= ml + ks * sigma:
        return np.exp(-0.5 * ((x - ml) / sigma) ** 2)
    else:
        return 0



def fouset(umf, lmf, xmin, xmax, xinc, yinc):
    S = [("FOU Points",)]  # to store points
    x = xmin
    while x <= xmax:
        upper = umf(x)
        lower = lmf(x)
        y = lower  # start from the lower value
        while y <= upper:
            if lower < y < upper:
                S.append((x, y))
            y += yinc
        x += xinc
    return S


def alpha_fouset(au, al, xinc, yinc):
    S = [("FOU Points",)]
    i = 1
    j = 0
    while j < min(len(al), len(au)):
        x = au[j][0] + xinc
        while x <= al[j][0] - xinc:
            S.append((x, au[j][2]))
            x += xinc
        x = al[j][1] + xinc
        while x <= au[j][1] - xinc:
            S.append((x, au[j][2]))
            x += xinc
        j += 1
    return S



# Parameters
# ml, mr, sigma, ks = -2, 2, 2, 3
# xmin, xmax, xinc, yinc = -10, 10, 0.01, 0.01

# # Generate FOU set
# x_coords, y_coords = fouset(partial(umf, ml=ml, mr=mr, sigma=sigma, ks=ks), 
#                            partial(lmf, ml=ml, mr=mr, sigma=sigma, ks=ks), 
#                            xmin, xmax, xinc, yinc)

# Create the plot
# plt.figure(figsize=(10, 6))
# plt.scatter(x_coords, y_coords, c='blue', s=1, label='FOU Points')  # s=1 for smaller point size
# plt.title('Fuzzy Output (FOU) Set of Points')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(True)
# plt.show()
