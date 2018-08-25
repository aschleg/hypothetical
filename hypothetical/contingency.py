# encoding=utf8

"""
Functions related to the analysis of contingency tables.

Contingency Tables
------------------

.. autosummary::
    :toctree: generated/

    ChiSquareContingency
    McNemarTest

Other Functions
---------------

.. autosummary::
    :toctree: generated/

    table_margin
    expected_frequencies

References
----------
Fagerland, M. W., Lydersen, S., & Laake, P. (2013).
    The McNemar test for binary matched-pairs data: Mid-p and asymptotic are better than exact conditional.
    Retrieved April 14, 2018, from http://www.biomedcentral.com/1471-2288/13/91

Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
    McGraw-Hill. ISBN 07-057348-4

Weisstein, Eric W. "Fisher's Exact Test." From MathWorld--A Wolfram Web Resource.
    http://mathworld.wolfram.com/FishersExactTest.html

Wikipedia contributors. (2018, May 20). Fisher's exact test. In Wikipedia, The Free Encyclopedia.
    Retrieved 12:46, August 14, 2018,
    from https://en.wikipedia.org/w/index.php?title=Fisher%27s_exact_test&oldid=842100719

"""

from functools import reduce
import numpy as np
from scipy.special import comb
from scipy.stats import chi2, binom


class ChiSquareContingency(object):
    r"""

    References
    ----------
    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    """
    def __init__(self, observed, expected=None, continuity=True):
        if not isinstance(observed, np.ndarray):
            self.observed = np.array(observed)
        else:
            self.observed = observed

        if expected is not None:
            if not isinstance(expected, np.ndarray):
                self.expected = np.array(expected)
            else:
                self.expected = expected

            if self.observed.shape != self.expected.shape:
                raise ValueError('observed and expected frequency contingency tables must have the same shape.')
        else:
            self.expected = expected_frequencies(self.observed)

        self.continuity = continuity
        self.degrees_freedom = (self.observed.shape[0] - 1) * (self.observed.shape[1] - 1)

        self.chi_square = self._chi_square()
        self.p_value = self._p_value()
        self.test_summary = {
            'chi-square': self.chi_square,
            'p-value': self.p_value,
            'degrees of freedom': self.degrees_freedom,
            'continuity': self.continuity
        }

    def _chi_square(self):
        cont_table = np.absolute(self.observed - self.expected)

        if self.degrees_freedom == 1:
            chi_val = np.sum((cont_table - (0.5 * self.continuity)) ** 2 / self.expected)
        else:
            chi_val = np.sum(cont_table ** 2 / self.expected)

        return chi_val

    def _p_value(self):
        pval = chi2.sf(self.chi_square, self.degrees_freedom)

        return pval


class FisherTest(object):
    r"""
    Performs Fisher's Exact Test for a 2x2 contingency table.

    Parameters
    ----------

    Attributes
    ----------

    Notes
    -----

    Examples
    --------

    References
    ----------
    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Weisstein, Eric W. "Fisher's Exact Test." From MathWorld--A Wolfram Web Resource.
        http://mathworld.wolfram.com/FishersExactTest.html

    Wikipedia contributors. (2018, May 20). Fisher's exact test. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:46, August 14, 2018,
        from https://en.wikipedia.org/w/index.php?title=Fisher%27s_exact_test&oldid=842100719

    """
    def __init__(self, table=None, alternative='two-sided'):
        if not isinstance(table, np.ndarray):
            self.table = np.array(table)
        else:
            self.table = table

        if self.table.shape != (2, 2):
            raise ValueError("Fisher's Exact Test requires a 2x2 contingency table with non-negative integers.")

        if (self.table < 0).any():
            raise ValueError('All values in table should be non-negative.')

        self.n = np.sum(self.table)
        self.p_value = self._p_value()
        self.odds_ratio = self._odds_ratio()
        self.test_summary = self._generate_test_summary()

    def _p_value(self):
        a, b, c, d = self.table[0, 0], self.table[0, 1], self.table[1, 0], self.table[1, 1]

        p = (comb(a + c, a) * comb(b + d, b)) / comb(self.n, a + b)

        return p

    def _odds_ratio(self):
        if self.table[1, 0] > 0 and self.table[0, 1] > 0:
            oddsratio = self.table[0, 0] * self.table[1, 1] / (self.table[1, 0] * self.table[0, 1])
        else:
            oddsratio = np.inf

        return oddsratio

    def _generate_test_summary(self):

        results = {
            'p-value': self.p_value,
            'odds ratio': self.odds_ratio,
            'contigency table': self.table
        }

        return results


class McNemarTest(object):
    r"""
    Computes the McNemar Test for two related samples in a 2x2 contingency table.

    Parameters
    ----------
    table : array-like
    continuity : bool, False
    alternative : str, {'two-sided', 'greater', 'less'}

    Attributes
    ----------
    table : array-like
    alternative : str
    continuity : bool
    n : int
    mcnemar_x2_statistic : float
    z_asymptotic_statistic : float
    mcnemar_p_value : float
    exact_p_value : float
    mid_p_value : float
    test_summary : dict

    Notes
    -----

    Examples
    --------

    References
    ----------
    Fagerland, M. W., Lydersen, S., & Laake, P. (2013).
        The McNemar test for binary matched-pairs data: Mid-p and asymptotic are better than exact conditional.
        Retrieved April 14, 2018, from http://www.biomedcentral.com/1471-2288/13/91

    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2018, April 29). McNemar's test. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:24, August 15, 2018,
        from https://en.wikipedia.org/w/index.php?title=McNemar%27s_test&oldid=838855782

    """
    def __init__(self, table, continuity=False, alternative='two-sided'):
        if not isinstance(table, np.ndarray):
            self.table = np.array(table)
        else:
            self.table = table

        if self.table.shape != (2, 2):
            raise ValueError("Fisher's Exact Test requires a 2x2 contingency table with non-negative integers.")

        if (self.table < 0).any():
            raise ValueError('All values in table should be non-negative.')

        if alternative not in ('two-sided', 'greater', 'less'):
            raise ValueError("alternative must be one of 'two-sided' (default), 'greater', or 'lesser'.")

        self.n = np.sum(self.table)
        self.continuity = continuity
        self.alternative = alternative

        self.mcnemar_x2_statistic = self._mcnemar_test_stat()
        self.z_asymptotic_statistic = self._asymptotic_test()
        self.mcnemar_p_value = self._mcnemar_p_value()
        self.exact_p_value = self._exact_p_value()
        self.mid_p_value = self._mid_p_value()
        self.test_summary = self._generate_test_summary()

    def _mcnemar_test_stat(self):

        if not self.continuity:
            x2 = (self.table[0, 1] - self.table[1, 0]) ** 2 / (self.table[0, 1] + self.table[1, 0])
        else:
            x2 = (np.absolute(self.table[0, 1] - self.table[1, 0]) - 1) ** 2 / (self.table[0, 1] + self.table[1, 0])

        return x2

    def _mcnemar_p_value(self):
        p = 1 - chi2.cdf(self.mcnemar_x2_statistic, 1)

        return p

    def _asymptotic_test(self):
        if not self.continuity:
            z_asymptotic = (self.table[1, 0] - self.table[0, 1]) / np.sqrt(self.table[0, 1] + self.table[1, 0])
        else:
            z_asymptotic = (np.absolute(self.table[1, 0] - self.table[0, 1]) - 1) / \
                           np.sqrt(self.table[0, 1] + self.table[1, 0])

        return z_asymptotic

    def _exact_p_value(self):
        i = self.table[0, 1]
        n = self.table[1, 0] + self.table[0, 1]
        i_n = np.arange(i + 1, n + 1)

        p_value = 1 - np.sum(comb(n, i_n) * 0.5 ** i_n * (1 - 0.5) ** (n - i_n))

        if self.alternative == 'two-sided':
            p_value *= 2

        return p_value

    def _mid_p_value(self):
        n = self.table[1, 0] + self.table[0, 1]
        mid_p = self.exact_p_value - binom.pmf(self.table[0, 1], n, 0.5)

        return mid_p

    def _generate_test_summary(self):

        results = {
            'N': self.n,
            'continuity': self.continuity,
            'alternative': self.alternative,
            'McNemar x2-statistic': self.mcnemar_x2_statistic,
            'Asymptotic z-statistic': self.z_asymptotic_statistic,
            'McNemar p-value': self.mcnemar_p_value,
            'Exact p-value': self.exact_p_value,
            'Mid p-value': self.mid_p_value
        }

        return results


def table_margins(table):
    r"""
    Computes the marginal sums of a one or two-dimensional array.

    """
    if not isinstance(table, np.ndarray):
        table = np.array(table).copy()

    if table.ndim > 2:
        raise ValueError('table must be a one or two-dimensional array.')

    table_dim = table.ndim

    c = np.apply_over_axes(np.sum, table, 0)

    if table_dim == 2:
        r = np.apply_over_axes(np.sum, table, 1)
    else:
        r = table

    return r, c


def expected_frequencies(observed):
    if not isinstance(observed, np.ndarray):
        observed = np.array(observed).copy()

    if observed.ndim > 2:
        raise ValueError('table dimension cannot be greater than two.')

    margins = table_margins(observed)

    exp_freq = reduce(np.multiply, margins) / np.sum(observed)

    return exp_freq
