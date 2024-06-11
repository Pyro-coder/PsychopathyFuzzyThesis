import fou_points
import trapezoid
import trapezoid as tpz
import support_interval as si
import alpha_cut_algorithm as aca
import alpha_cuts_UMF_LMF as acul
import fou_points as fp
import alpha_cut_power_mean as acpm

import numpy as np
import pandas as pd


def very_bad_lower(x):
    return tpz.trap(x, 1, 0, 0, 1.5, 3.707)


def very_bad_upper(x):
    return tpz.trap(x, 1, 0, 0, 1.5, 3.997)


def bad_lower(x):
    return tpz.trap(x, 1, 1.255, 2, 2.5, 4.625)


def bad_upper(x):
    return tpz.trap(x, 1, 0, 2, 2.5, 4.944)


def somewhat_bad_lower(x):
    return tpz.trap(x, 1, 1.5, 3, 3.5, 4.99)


def somewhat_bad_upper(x):
    return tpz.trap(x, 1, 0.957, 3, 3.5, 6.11)


def fair_lower(x):
    return tpz.trap(x, 1, 2.907, 5, 5.5, 7.717)


def fair_upper(x):
    return tpz.trap(x, 1, 2.893, 5, 5.5, 7.756)


def somewhat_good_lower(x):
    return tpz.trap(x, 1, 4.835, 6.5, 6.75, 8.623)


def somewhat_good_upper(x):
    return tpz.trap(x, 1, 4.595, 6.5, 6.75, 7.5)


def good_lower(x):
    return tpz.trap(x, 1, 5.409, 7, 7.5, 8.577)


def good_upper(x):
    return tpz.trap(x, 1, 4.648, 7, 7.5, 9.616)


def very_good_lower(x):
    return tpz.trap(x, 1, 7.944, 8, 10, 10)


def very_good_upper(x):
    return tpz.trap(x, 1, 6.172, 8, 10, 10)


def antonym(f, x):
    return f(10 - x)


def complement(f, x):
    return 1 - f(x)


def z7_lower(x):
    """
    Array of LMFs for 7-word vocabulary.
    Args:
        x: The input to the functions.
    Returns:
        A NumPy array containing the results of the functions applied to x.
    """
    return np.array([
        very_bad_lower(x),
        bad_lower(x),
        somewhat_bad_lower(x),
        fair_lower(x),
        somewhat_good_lower(x),
        good_lower(x),
        very_good_lower(x)
    ])


def z7_upper(x):
    """
    Array of UMFs for 7-word vocabulary.
    Args:
        x: The input to the functions.
    Returns:
        A NumPy array containing the results of the functions applied to x.
    """
    return np.array([
        very_bad_upper(x),
        bad_upper(x),
        somewhat_bad_upper(x),
        fair_upper(x),
        somewhat_good_upper(x),
        good_upper(x),
        very_good_upper(x)
    ])


# next, compute the alpha cuts for each of the vocabulary words

def f_very_bad(x):
    return np.array([z7_upper(x)[0], z7_lower(x)[0]])


def f_bad(x):
    return np.array([z7_upper(x)[1], z7_lower(x)[1]])


def f_somewhat_bad(x):
    return np.array([z7_upper(x)[2], z7_lower(x)[2]])


def f_fair(x):
    return np.array([z7_upper(x)[3], z7_lower(x)[3]])


def f_somewhat_good(x):
    return np.array([z7_upper(x)[4], z7_lower(x)[4]])


def f_good(x):
    return np.array([z7_upper(x)[5], z7_lower(x)[5]])


def f_very_good(x):
    return np.array([z7_upper(x)[6], z7_lower(x)[6]])


alpha_very_bad = acpm.alpha_t2(f_very_bad, 1000, 0, 10, 100, 300)
alpha_bad = acpm.alpha_t2(f_bad, 1000, 0, 10, 100, 300)
alpha_somewhat_bad = acpm.alpha_t2(f_somewhat_bad, 1000, 0, 10, 100, 300)
alpha_fair = acpm.alpha_t2(f_fair, 1000, 0, 10, 100, 300)
alpha_somewhat_good = acpm.alpha_t2(f_somewhat_good, 1000, 0, 10, 100, 300)
alpha_good = acpm.alpha_t2(f_good, 1000, 0, 10, 100, 300)
alpha_very_good = acpm.alpha_t2(f_very_good, 1000, 0, 10, 100, 300)


def alpha_antonym(a):
    """
    Perform transformations on the input array `a` according to the specified logic.

    Args:
        a: A NumPy array with at least 3 columns.

    Returns:
        out: A NumPy array with the transformed values.
    """
    rows, cols = a.shape
    out = np.copy(a)

    for i in range(rows):
        out[i, 1] = 10 - a[i, 0]
        out[i, 0] = 10 - a[i, 1]
        out[i, 2] = a[i, 2]

    return out


alpha_7 = np.array([alpha_very_bad, alpha_bad, alpha_somewhat_bad, alpha_fair, alpha_somewhat_good, alpha_good,
                    alpha_very_good])

file_path = "excel/PCLRWords.xlsx"
words = pd.read_excel(file_path, sheet_name="Words", usecols="A:H", skiprows=2, nrows=20, header=None)

weight_full_vocab = pd.read_excel(file_path, sheet_name="Words", usecols="B:E", skiprows=26, nrows=1, header=None)

score_sheet = pd.read_excel(file_path, sheet_name="Scores", usecols="A:C", nrows=21)

scores = score_sheet['Scoring']
trait_weights = score_sheet['Weights']

vocab = score_sheet['Factors']
vocab = vocab.to_frame()
vocab['Vocab'] = 7


def create_scoredata():
    out = pd.DataFrame(index=score_sheet['Factors'], columns=['Alpha Cuts'])
    for i in range(20):
        for j in range(vocab.iloc[i, 1]):
            if scores[i] == words.at[i, j + 1]:
                out.at[score_sheet['Factors'][i], 'Alpha Cuts'] = alpha_7[j]
    return out


score_data = create_scoredata()
print(score_data)


# Define 4 word vocabulary

def Ul(x):
    return trapezoid.trap(x, 1, 0, 0, 1, 3)


def Uu(x):
    return trapezoid.trap(x, 1, 0, 0, 1, 5.96)


def MLUl(x):
    return trapezoid.trap(x, 1, 1.98, 3, 3.5, 4.6)


def MLUu(x):
    return trapezoid.trap(x, 1, 0, 3, 3.5, 6.32)


def MLIl(x):
    return trapezoid.trap(x, 1, 4.62, 6.1, 6.5, 7.69)


def MLIu(x):
    return trapezoid.trap(x, 1, 3.98, 6.1, 6.5, 9.04)


def VIl(x):
    return trapezoid.trap(x, 1, 6.77, 8, 10, 10)


def VIu(x):
    return trapezoid.trap(x, 1, 4.52, 8, 10, 10)


Ufou = fou_points.fouset(Uu, Ul, 0, 10, 0.1, 0.035)
MLUfou = fou_points.fouset(MLUu, MLUl, 0, 10, 0.09, 0.025)
MLIfou = fou_points.fouset(MLIu, MLIl, 0, 10, 0.011, 0.0355)
VIfou = fou_points.fouset(VIu, VIl, 0, 10, 0.095, 0.0255)


def z4_lower(x):
    return np.array([Ul(x), MLUl(x), MLIl(x), VIl(x)])


def z4_upper(x):
    return np.array([Uu(x), MLUu(x), MLIu(x), VIu(x)])


def f_U(x):
    return np.array([z4_upper(x)[0], z4_lower(x)[0]])


def f_MLU(x):
    return np.array([z4_upper(x)[1], z4_lower(x)[1]])


def f_MLI(x):
    return np.array([z4_upper(x)[2], z4_lower(x)[2]])


def f_VI(x):
    return np.array([z4_upper(x)[3], z4_lower(x)[3]])


alpha_U = acpm.alpha_t2(f_U, 1000, 0, 10, 100, 300)
alpha_MLU = acpm.alpha_t2(f_MLU, 1000, 0, 10, 100, 300)
alpha_MLI = acpm.alpha_t2(f_MLI, 1000, 0, 10, 100, 300)
alpha_VI = acpm.alpha_t2(f_VI, 1000, 0, 10, 100, 300)

alpha_4 = np.array([alpha_U, alpha_MLU, alpha_MLI, alpha_VI])

vocab_4 = score_sheet['Weights']
vocab_4 = vocab_4.to_frame()
vocab_4['Vocab'] = 4

# TODO: Troubleshoot the following
def create_weightdata():
    out = pd.DataFrame(index=score_sheet['Weights'], columns=['Alpha Cuts'])
    for i in range(20):
        for j in range(vocab_4.iloc[i, 1]):
            if trait_weights[i] == weight_full_vocab.at[j + 1, 0]:
                out.at[score_sheet['Weights'][i], 'Alpha Cuts'] = alpha_4[j]
    return out


weight_data = create_weightdata()
print(weight_data)
