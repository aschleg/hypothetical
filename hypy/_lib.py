import numpy as np
from scipy.stats import t


def _t_conf_int(x, dof, y=None):

    xbar, xn, xvar = x[0], x[1], x[2]

    if y is not None:
        ybar, yn, yvar = y[0], y[1], y[2]

        low_interval = (xbar - ybar) + t.ppf(0.025, dof) * np.sqrt(xvar / xn + yvar / yn)
        high_interval = (xbar - ybar) - t.ppf(0.025, dof) * np.sqrt(xvar / xn + yvar / yn)
    else:
        low_interval = xbar + 1.96 * np.sqrt((xbar * (1 - xbar)) / xn)
        high_interval = xbar - 1.96 * np.sqrt((xbar * (1 - xbar)) / xn)

    return float(low_interval), float(high_interval)


def _student_t_pvalue(n, dof, test='two-tail'):
    p = (1. - t.cdf(n, dof))

    if test == 'two-tail':
        p *= 2.

    return p
