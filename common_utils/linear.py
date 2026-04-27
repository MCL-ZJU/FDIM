from scipy.stats import linregress
import numpy as np


def func_linear(x, b, extra=0.0):
    """
    y = b_1 * x + b_0
    """
    x = np.array(x, dtype=np.float64)
    return b[1] * x + b[0] - extra


def fit(x, y, m_min=0, m_max=100):

    res = linregress(x, y)

    if res is None:
    # if not res.success:
        print(res)
        raise ValueError('optimization failed')
    coef = np.array([res.intercept, res.slope])
    return coef