import numpy as np


def degrees_of_freedom(y1, y2=None, var_equal=False):
    r"""
    Computes the degrees of freedom of one or two samples.

    Parameters
    ----------
    y1
        First sample to test
    y2
        Second sample. Optional.
    var_equal
        Optional, default False. If False, Welch's t-test for unequal variance and
        sample sizes is used. If True, equal variance between samples is assumed
        and Student's t-test is used.

    Returns
    -------
    float
        the degrees of freedom

    Notes
    -----
    When Welch's t test is used, the Welch-Satterthwaite equation for approximating the degrees
    of freedom should be used and is defined as:

    .. math::

        \large v \approx \frac{\left(\frac{s_{1}^2}{N_1} +
        \frac{s_{2}^2}{N_2}\right)^2}{\frac{\left(\frac{s_1^2}{N_1^{2}}\right)^2}{v_1} +
        \frac{\left(\frac{s_2^2}{N_2^{2}}\right)^2}{v_2}}

    If the two samples are assumed to have equal variance, the degrees of freedoms become simply:

    .. math::

        v = n_1 + n_2 - 2

    In the case of one sample, the degrees of freedom are:

    .. math::

        v = n - 1

    References
    ----------
    Rencher, A. C., & Christensen, W. F. (2012). Methods of multivariate analysis (3rd Edition).

    Welch's t-test. (2017, June 16). In Wikipedia, The Free Encyclopedia.
        From https://en.wikipedia.org/w/index.php?title=Welch%27s_t-test&oldid=785961228

    """
    y1 = np.array(y1)
    n1 = len(y1)
    s1 = np.var(y1)
    v1 = n1 - 1

    if y2 is not None:
        y2 = np.array(y2)
        n2 = len(y2)
        s2 = np.var(y2)
        v2 = n2 - 1

        if var_equal is False:
            v = np.power((s1 / n1 + s2 / n2), 2) / (np.power((s1 / n1), 2) / v1 + np.power((s2 / n2), 2) / v2)
        else:
            v = n1 + n2 - 2

    else:
        v = v1

    return float(v)