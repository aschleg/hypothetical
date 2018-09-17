
"""

Implementations of goodness-of-fit tests.

Goodness-of-fit
---------------

.. autosummary::
    :toctree: generated/

    ChiSquareTest
    jarque_bera

References
----------
B. W. Yap & C. H. Sim (2011) Comparisons of various types of normality tests,
    Journal of Statistical Computation and Simulation, 81:12, 2141-2155, DOI: 10.1080/00949655.2010.520163

Ukponmwan H. Nosakhare, Ajibade F. Bright. Evaluation of Techniques for Univariate Normality Test Using Monte
    Carlo Simulation. American Journal of Theoretical and Applied Statistics.
    Special Issue: Statistical Distributions and Modeling in Applied Mathematics.
    Vol. 6, No. 5-1, 2017, pp. 51-61. doi: 10.11648/j.ajtas.s.2017060501.18

Wikipedia contributors. (2018, March 20). Jarque–Bera test. In Wikipedia, The Free Encyclopedia.
    Retrieved 14:46, September 15, 2018,
    from https://en.wikipedia.org/w/index.php?title=Jarque%E2%80%93Bera_test&oldid=831439673

"""
import numpy as np
from scipy.stats import chi2
from hypothetical.descriptive import kurtosis, skewness


class ChiSquareTest(object):
    r"""
    Performs the one-sample Chi-Square goodness-of-fit test.

    Parameters
    ----------
    observed : array-like
        One-dimensional array of observed frequencies.
    expected : array-like, optional
        One-dimensional array of expected frequencies. If not given, the expected frequencies are computed
        as the mean of the observed frequencies (each category is equally likely to occur).
    continuity : bool, optional
        Applies Yates's continuity correction for approximation error. Defaults to False as the correction can
        tend to overcorrect and result in a type II error.
    degrees_freedom : int, optional
        Degrees of freedom. The p-value in the chi-square test is computed with degrees of freedom is :math:`k - 1`,
        where :math:`k` is the number of observed frequencies.

    Attributes
    ----------
    observed : array-like
        One-dimensional array of observed frequencies.
    expected : array-like
        One-dimensional array of expected frequencies.
    degrees_of_freedom : int
        Total degrees of freedom used in the computation of the p-value.
    continuity_correction : bool
        If True, Yates's continuity correction is applied when performing the chi-square test
    n : int
        Total number of observed frequencies.
    chi_square : float
        The computed test statistic, the chi-square, :math:`\chi^2` value.
    p_value : float
        The calculated p-value of the test given the chi-square statistic and degrees of freedom.
    test_summary : dict
        Dictionary containing a collection of the resulting test statistics and other information.

    Raises
    ------
    ValueError
        If the :code:`expected` parameter is passed but is not of the same length as the required :code:`observed`
        parameter, a :code:`ValueError` is raised.

    Notes
    -----
    The chi-squared test, often called the :math:`\chi^2` test, is also known as Pearson's chi-squared test. The
    chi-square test is a one-sample goodnes-of-fit test that evaluates whether a significant difference exists between
    an observed number of frequencies from two or more groups and an expected frequency based on a null hypothesis. A
    simple example of a chi-square test is testing wheher a six-sided die is 'fair', in that all outcomes are equally
    likely to occur.

    The chi-square test statistic, :math:`\chi^2` is defined as:

    .. math::

        \chi^2 = \sum^k_{i=1} \frac{O_i - E_i)^2}{E_i}

    Where :math:`O_i` is the observed number of frequencies in the :math:`i`th category, :math:`E_i` is the expected
    number of frequencies in the respective :math:`i`th group, and :math:`k` is the total number of groups or
    categories, or 'cells'.

    The p-value can then be found by comparing the calculated :math:`\chi^2` statistic to a chi-square distribution.
    The degrees of freedom is equal to :math:`k - 1` minus any additional reduction in the degrees of freedom, if
    specified.

    Examples
    --------
    >>> observed = [29, 19, 18, 25, 17, 10, 15, 11]
    >>> expected = [18, 18, 18, 18, 18, 18, 18, 18]
    >>> ch = ChiSquareTest(observed, expected)
    >>> ch.test_summary
    {'chi-square': 16.333333333333332,
     'continuity correction': False,
     'degrees of freedom': 7,
     'p-value': 0.022239477462390588}
    >>> ch = ChiSquareTest(observed)
    >>> ch.test_summary
    {'chi-square': 16.333333333333332,
     'continuity correction': False,
     'degrees of freedom': 7,
     'p-value': 0.022239477462390588}

    References
    ----------
    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Weisstein, Eric W. "Chi-Squared Test." From MathWorld--A Wolfram Web Resource.
        http://mathworld.wolfram.com/Chi-SquaredTest.html

    Wikipedia contributors. (2018, July 5). Chi-squared test. In Wikipedia, The Free Encyclopedia. Retrieved 13:56,
        August 19, 2018, from https://en.wikipedia.org/w/index.php?title=Chi-squared_test&oldid=848986171

    Wikipedia contributors. (2018, April 12). Pearson's chi-squared test. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:55, August 23, 2018,
        from https://en.wikipedia.org/w/index.php?title=Pearson%27s_chi-squared_test&oldid=836064929

    """
    def __init__(self, observed, expected=None, continuity=False, degrees_freedom=0):

        if not isinstance(observed, np.ndarray):
            self.observed = np.array(observed)
        else:
            self.observed = observed

        if expected is None:
            obs_mean = np.mean(self.observed)
            self.expected = np.full_like(self.observed, obs_mean)

        else:
            if not isinstance(expected, np.ndarray):
                self.expected = np.array(expected)
            else:
                self.expected = expected

            if self.observed.shape[0] != self.expected.shape[0]:
                raise ValueError('number of observations must be of the same length as expected values.')

        self.degrees_of_freedom = self.observed.shape[0] - 1 - degrees_freedom
        self.continuity_correction = continuity
        self.n = self.observed.shape[0]
        self.chi_square = self._chisquare_value()
        self.p_value = self._p_value()
        self.test_summary = {
            'chi-square': self.chi_square,
            'p-value': self.p_value,
            'degrees of freedom': self.degrees_of_freedom,
            'continuity correction': self.continuity_correction
        }

    def _chisquare_value(self):
        x2 = np.sum((np.absolute(self.observed - self.expected) - (0.5 * self.continuity_correction)) ** 2 /
                    self.expected)

        return x2

    def _p_value(self):
        pval = chi2.sf(self.chi_square, self.degrees_of_freedom)

        return pval


def jarque_bera(x):
    r"""
    Performs the Jarque-Bera goodness-of-fit test.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    test_result : dict

    Examples
    --------

    Notes
    -----

    References
    ----------
    B. W. Yap & C. H. Sim (2011) Comparisons of various types of normality tests,
        Journal of Statistical Computation and Simulation, 81:12, 2141-2155, DOI: 10.1080/00949655.2010.520163

    Ukponmwan H. Nosakhare, Ajibade F. Bright. Evaluation of Techniques for Univariate Normality Test Using Monte
        Carlo Simulation. American Journal of Theoretical and Applied Statistics.
        Special Issue: Statistical Distributions and Modeling in Applied Mathematics.
        Vol. 6, No. 5-1, 2017, pp. 51-61. doi: 10.11648/j.ajtas.s.2017060501.18

    Wikipedia contributors. (2018, March 20). Jarque–Bera test. In Wikipedia, The Free Encyclopedia.
        Retrieved 14:46, September 15, 2018,
        from https://en.wikipedia.org/w/index.php?title=Jarque%E2%80%93Bera_test&oldid=831439673

    """
    n = len(x)

    jb = n / 6. * (skewness(x) ** 2 + kurtosis(x) ** 2 / 4)

    p_value = chi2.sf(jb, 2)

    test_result = {
        'JB test statistic': jb,
        'p-value': p_value
    }

    return test_result
