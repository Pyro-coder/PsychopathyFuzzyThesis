from max_alpha_for_lmf import alpha_maxlmf
from alpha_cut_algorithm import alpha_cut
from EKM_algorithm import selectf, kmalg

import numpy as np

def alpha_fwa(x, w, xsup, wsup, m, n):
    num_terms = xsup.shape[0]
    xalpha = []
    walpha = []
    
    for i in range(num_terms):
        fx = lambda y: selectf(x, y, i)
        fw = lambda y: selectf(w, y, i)
        
        xalpha.append(alpha_cut(fx, xsup[i, 0], xsup[i, 1], m, n))
        walpha.append(alpha_cut(fw, wsup[i, 0], wsup[i, 1], m, n))
    
    hmin = alpha_maxlmf(xalpha, walpha)
    
    xab = [None] * (hmin[0] + 1)
    wcd = [None] * (hmin[0] + 1)
    
    for j in range(hmin[0] + 1):
        for i in range(num_terms):
            if xab[j] is None:
                xab[j] = np.array([xalpha[i][j]])
            else:
                xab[j] = np.vstack((xab[j], xalpha[i][j]))
            
            if wcd[j] is None:
                wcd[j] = np.array([walpha[i][j]])
            else:
                wcd[j] = np.vstack((wcd[j], walpha[i][j]))
    
    out = None
    for j in range(hmin[0] + 1):
        z = kmalg(xab[j], wcd[j])
        if isinstance(z, tuple):
            z = np.array(z)  # Convert to a NumPy array if it's a tuple
        if j == 0:
            out = np.array([z[0], z[1], xab[j][0, 2]])
        else:
            out = np.vstack((out, [z[0], z[1], xab[j][0, 2]]))
    
    return out