from collections import namedtuple

import numpy as np

from hypy._lib import _student_t_pvalue, _t_conf_int
from hypy.dof import degrees_of_freedom


class tTest(object):

    def __init__(self, y1, y2=None, mu=None, var_equal=False, paired=False):
        pass



def ttest(y1, y2=None, mu=None, var_equal=False):
    r"""
    Performs one and two-sample t-tests.

    Parameters
    ----------
    y1
        First sample to test
    y2
        Second sample. Optional
    mu
        Optional, sets the mean for comparison in the one sample t-test. Default 0.
    var_equal
        Optional, default False. If False, Welch's t-test for unequal variance and
        sample sizes is used. If True, equal variance between samples is assumed
        and Student's t-test is used.

    Returns
    -------
    namedtuple
        Namedtuple containing following values:
        t-value
        degrees of freedom
        p-value
        confidence intervals
        sample means

    Notes
    -----
    Welch's t-test is an adaption of Student's t test and is more performant when the
    sample variances and size are unequal. The test still depends on the assumption of
    the underlying population distributions being normally distributed.

    Welch's t test is defined as:

    .. math::

        t = \frac{\bar{X_1} - \bar{X_2}}{\sqrt{\frac{s_{1}^{2}}{N_1} + \frac{s_{2}^{2}}{N_2}}}

    where:

    :math:`\bar{X}` is the sample mean, :math:`s^2` is the sample variance, :math:`n` is the sample size

    If the :code:`var_equal` argument is True, Student's t-test is used, which assumes the two samples
    have equal variance. The t statistic is computed as:

    .. math::

        t = \frac{\bar{X}_1 - \bar{X}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}

    where:

    .. math::

        s_p = \sqrt{\frac{(n_1 - 1)s^2_{X_1} + (n_2 - 1)s^2_{X_2}}{n_1 + n_2 - 2}

    References
    ----------
    Rencher, A. C., & Christensen, W. F. (2012). Methods of multivariate analysis (3rd Edition).

    Student's t-test. (2017, June 20). In Wikipedia, The Free Encyclopedia.
        From https://en.wikipedia.org/w/index.php?title=Student%27s_t-test&oldid=786562367

    """
    y1 = np.array(y1)

    n1 = len(y1)
    s1 = np.var(y1)
    ybar1 = np.mean(y1)

    if y2 is not None:
        y2 = np.array(y2)
        n2 = len(y2)
        s2 = np.var(y2)
        ybar2 = np.mean(y2)

        if var_equal is False:
            tval = float((ybar1 - ybar2) / np.sqrt(s1 / n1 + s2 / n2))
        else:
            sp = np.sqrt(((n1 - 1.) * s1 + (n2 - 1.) * s2) / (n1 + n2 - 2.))
            tval = float((ybar1 - ybar2) / (sp * np.sqrt(1. / n1 + 1. / n2)))

    else:
        ybar2, n2, s2 = 0.0, 1.0, 0.0
        if mu is None:
            mu = 0.0

        tval = float((ybar1 - mu) / np.sqrt(s1 / n1))

    dof = degrees_of_freedom(y1, y2)
    pvalue = _student_t_pvalue(np.absolute(tval), dof)
    intervals = _t_conf_int((ybar1, n1, s1), dof=dof, y=(ybar2, n2, s2))

    if y2 is not None:
        tTestResult = namedtuple('tTestResult', ['tvalue', 'dof', 'pvalue', 'confint', 'x_mean', 'y_mean'])

        tt = tTestResult(tvalue=tval, dof=dof, pvalue=pvalue, confint=intervals, x_mean=ybar1, y_mean=ybar2)

    else:
        tTestResult = namedtuple('tTestResult', ['tvalue', 'dof', 'pvalue', 'confint', 'x_mean'])
        tt = tTestResult(tvalue=tval, dof=dof, pvalue=pvalue, confint=intervals, x_mean=ybar1)

    return tt