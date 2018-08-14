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
Weisstein, Eric W. "Fisher's Exact Test." From MathWorld--A Wolfram Web Resource.
        http://mathworld.wolfram.com/FishersExactTest.html

Wikipedia contributors. (2018, May 20). Fisher's exact test. In Wikipedia, The Free Encyclopedia.
    Retrieved 12:46, August 14, 2018,
    from https://en.wikipedia.org/w/index.php?title=Fisher%27s_exact_test&oldid=842100719

"""

import numpy as np
from scipy.special import comb


class ContingencyTable(object):

    def __init__(self):
        pass


class FisherTest(object):
    r"""

    References
    ----------
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

    def __init__(self):
        pass
