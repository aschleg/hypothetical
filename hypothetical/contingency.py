# encoding=utf8

"""
Functions related to the analysis of contingency tables.

Contingency Tables
------------------

.. autosummary::
    :toctree: generated/

    FisherTest

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

import numpy as np
from scipy.special import comb
from scipy.stats import chi2


class ContingencyTable(object):

    def __init__(self):
        pass


class FisherTest(object):
    r"""
    Performs Fisher's Exact Test for a 2x2 contingency table.

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
    Computes the McNemar Test for two related samples.

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
    Fagerland, M. W., Lydersen, S., & Laake, P. (2013).
        The McNemar test for binary matched-pairs data: Mid-p and asymptotic are better than exact conditional.
        Retrieved April 14, 2018, from http://www.biomedcentral.com/1471-2288/13/91

    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2018, April 29). McNemar's test. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:24, August 15, 2018,
        from https://en.wikipedia.org/w/index.php?title=McNemar%27s_test&oldid=838855782

    """
    def __init__(self, table=None, continuity=False, alternative='two-sided'):
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

    def _mcnemar_test_stat(self):

        if not self.continuity:
            x2 = (self.table[0, 1] - self.table[1, 0]) ** 2 / (self.table[0, 1] + self.table[1, 0])
        else:
            x2 = (np.absolute(self.table[0, 1] - self.table[1, 0]) - 1) ** 2 / (self.table[0, 1] + self.table[1, 0])

        return x2

    def _mcnemar_p_value(self):
        p = chi2.cdf(self.mcnemar_x2_statistic, 1)

        return p

    def _asymptotic_test(self):
        if not self.continuity:
            z_asymptotic = (self.table[0, 1] - self.table[1, 0]) / np.sqrt(self.table[0, 1] + self.table[1, 0])
        else:
            z_asymptotic = (np.absolute(self.table[0, 1] - self.table[1, 0]) - 1) / \
                           np.sqrt(self.table[0, 1] + self.table[1, 0])

        return z_asymptotic

    def _exact_p_value(self):
        i = self.table[0, 1]
        i_n = np.arange(i, self.n + 1)

        p_value = np.sum(comb(self.n, i_n) * 0.5 ** i_n * (1 - 0.5) ** (self.n - i_n))

        if self.alternative == 'two-sided':
            p_value *= 2

        return p_value

    def _mid_p_value(self):
        mid_p = self.exact_p_value - comb(self.n, self.table[0, 1]) * \
                0.5 ** self.table[0, 1] * (1 - 0.5) ** (self.n - self.table[0, 1])

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
