
"""

Implementations of goodness-of-fit tests.

Goodness-of-fit
---------------

.. autosummary::
    :toctree: generated/

    ChiSquareTest
    JarqueBera

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
    chi-square test is a one-sample goodness-of-fit test that evaluates whether a significant difference exists between
    an observed number of frequencies from two or more groups and an expected frequency based on a null hypothesis. A
    simple example of a chi-square test is testing whether a six-sided die is 'fair', in that all outcomes are equally
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
        r"""
        Computes the chi-square value of the sample data

        Notes
        -----
        The chi-square test statistic, :math:`\chi^2` is defined as:

        .. math::

            \chi^2 = \sum^k_{i=1} \frac{O_i - E_i)^2}{E_i}

        Returns
        -------
        x2 : float
            The computed chi-square value with continuity correction (if specified)

        """
        x2 = np.sum((np.absolute(self.observed - self.expected) - (0.5 * self.continuity_correction)) ** 2 /
                    self.expected)

        return x2

    def _p_value(self):
        r"""
        Finds the p-value of the chi-square statistic.

        Notes
        -----
        The p-value can be found by comparing the calculated :math:`\chi^2` statistic to a chi-square distribution.
        The degrees of freedom is equal to :math:`k - 1` minus any additional reduction in the degrees of freedom, if
        specified.

        Returns
        -------
        p_value : float
            The p-value of the associated chi-square value and degrees of freedom.

        """
        pval = chi2.sf(self.chi_square, self.degrees_of_freedom)

        return pval


class JarqueBera(object):
    r"""
    Performs the Jarque-Bera goodness-of-fit test.

    Parameters
    ----------
    x : array-like
        One-dimensional array-like object (list, numpy array, pandas DataFrame or pandas Series) containing
        the observed sample values.

    Attributes
    ----------
    x : array-like
        The given sample values
    test_statistic : float
        Computed Jarque-Bera test statistic
    p_value : float
        p-value of Jarque-Bera test statistic
    test_summary : dict
        Dictionary containing the Jarque-Bera test statistic and associated p-value.

    Examples
    --------

    Notes
    -----
    The Jarque-Bera test is a goodness-of-fit test developed by Carlos Jarque and Anil Bera that tests whether
    a sample of data is normally distributed using the sample's kurtosis and skewness. The Jarque-Bera test
    statistic is defined as:

    .. math::

        JB = \frac{n}{6} \large( s^2 + \frac{(k-3)^2}{4} \large)

    where :math:`n` is the number of samples in the data, :math:`s` is the computed sample's skewness and :math:`k` is
    the sample's kurtosis. The Jarque-Bera test statistic has a chi-square distribution with two degrees of freedom
    when the number of samples is adequate. The test statistic is always non-negative and the farther away from zero,
    the stronger of an indication the sample data does not follow a normal distribution.

    In the case of small samples ('small' being somewhat subjective but generally considered to be :math:`n < 30`),
    the Jarque-Bera test and statistic is overly-sensitive and can lead to large Type 1 error rates.

    References
    ----------
    B. W. Yap & C. H. Sim (2011) Comparisons of various types of normality tests,
        Journal of Statistical Computation and Simulation, 81:12, 2141-2155, DOI: 10.1080/00949655.2010.520163

    Jarque, C., & Bera, A. (1987). A Test for Normality of Observations and Regression Residuals.
        International Statistical Review / Revue Internationale De Statistique, 55(2), 163-172. doi:10.2307/1403192

    Ukponmwan H. Nosakhare, Ajibade F. Bright. Evaluation of Techniques for Univariate Normality Test Using Monte
        Carlo Simulation. American Journal of Theoretical and Applied Statistics.
        Special Issue: Statistical Distributions and Modeling in Applied Mathematics.
        Vol. 6, No. 5-1, 2017, pp. 51-61. doi: 10.11648/j.ajtas.s.2017060501.18

    Wikipedia contributors. (2018, March 20). Jarque–Bera test. In Wikipedia, The Free Encyclopedia.
        Retrieved 14:46, September 15, 2018,
        from https://en.wikipedia.org/w/index.php?title=Jarque%E2%80%93Bera_test&oldid=831439673

    """
    def __init__(self, x):

        if not isinstance(x, np.ndarray):
            self.x = np.array(x)
        else:
            self.x = x

        if self.x.ndim != 1:
            raise ValueError('sample data must be one-dimensional')

        self.test_statistic = self._jarque_bera_statistic()
        self.p_value = self._p_value()
        self.test_summary = {
            'Jarque-Bera statistic': self.test_statistic,
            'p-value': self.p_value
        }

    def _jarque_bera_statistic(self):
        r"""
        Computes the Jarque-Bera test statistic:

        Returns
        -------
        jb : float
            The Jarque-Bera test statistic.

        Notes
        -----
        The Jarque-Bera test statistic is defined as:

        .. math::

            JB = \frac{n}{6} \large( s^2 + \frac{(k-3)^2}{4} \large)

        """
        n = len(self.x)

        jb = n / 6. * (skewness(self.x) ** 2 + kurtosis(self.x) ** 2 / 4)

        return jb

    def _p_value(self):
        r"""
        Calculates the associated p-value of the Jarque-Bera test statistic.

        Returns
        -------
        p_value : float
            The p-value of the Jarque-Bera test statistic.

        Notes
        -----
        The Jarque-Bera test statistic has a chi-square distribution with two degrees of freedom
        when the number of samples is adequate. The test statistic is always non-negative and the farther away from
        zero, the stronger of an indication the sample data does not follow a normal distribution.

        In the case of small samples ('small' being somewhat subjective but generally considered to be :math:`n < 30`),
        the Jarque-Bera test and statistic is overly-sensitive and can lead to large Type 1 error rates.

        """
        p_value = chi2.sf(self.test_statistic, 2)

        return p_value


class KolmogorovSmirnov(object):

    def __init__(self):
        pass
