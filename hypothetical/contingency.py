# encoding=utf8

"""
Functions related to the analysis of contingency tables.

Contingency Tables
------------------

.. autosummary::
    :toctree: generated/

    ChiSquareContingency
    CochranQ
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

Gibbons, J. D., & Chakraborti, S. (2010). Nonparametric statistical inference. London: Chapman & Hall.

Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
    McGraw-Hill. ISBN 07-057348-4

Weisstein, Eric W. "Fisher's Exact Test." From MathWorld--A Wolfram Web Resource.
    http://mathworld.wolfram.com/FishersExactTest.html

Wikipedia contributors. (2017, August 8). Cochran's Q test. In Wikipedia, The Free Encyclopedia.
    Retrieved 15:05, August 26, 2018,
    from https://en.wikipedia.org/w/index.php?title=Cochran%27s_Q_test&oldid=794571272

Wikipedia contributors. (2018, May 20). Fisher's exact test. In Wikipedia, The Free Encyclopedia.
    Retrieved 12:46, August 14, 2018,
    from https://en.wikipedia.org/w/index.php?title=Fisher%27s_exact_test&oldid=842100719

"""

from functools import reduce
import numpy as np
import numpy_indexed as npi
import pandas as pd
from scipy.special import comb
from scipy.stats import chi2, binom

from hypothetical._lib import build_des_mat


class ChiSquareContingency(object):
    r"""
    Performs the Chi-square test of independence of variables in an r x c table.

    Parameters
    ----------
    observed : array-like
        One-dimensional array-like object (list, numpy array, pandas DataFrame or pandas Series) containing
        the observed sample values.
    expected : array-like, optional
        One-dimensional array-like object (list, numpy array, pandas DataFrame or pandas Series) containing
        the observed sample values. If not passed, the expected frequencies are calculated using the
        :code:`expected_frequencies` function.
    continuity : bool, optional
        If True and degrees of freedom is equal to 1, Yates's continuity correction is applied.

    Attributes
    ----------
    observed : array-like
        The passed observation vector
    expected : array-like
        The passed expected frequencies vector.
    continuity : bool
        If True and degrees of freedom is equal to 1, Yates's continuity correction is applied.
    degrees_freedom : int
        Degrees of freedom, calculated as :math:`dof = (k - 1)(r - 1)` where :math:`k` is the number of columns
        and :math:`r` is the number of rows in the contingency table.
    chi_square : float
        The calculated chi-square value.
    p_value : float
        The associated p-value.
    association_measures : dict
        A dictionary containing the phi-coefficient, C, and Cramer's V association measures.
    test_summary : dict
        A dictionary containing the relevant test results.

    Raises
    ------
    ValueError
        If the observed and expected arrays are not of the same shape (if an expected array is passed).

    Examples
    --------
    >>> observed = [[23, 40, 16, 2], [11, 75, 107, 14], [1, 31, 60, 10]]
    >>> expected = [[7.3, 30.3, 38.0, 5.4], [18.6, 77.5, 97.1, 13.8], [9.1, 38.2, 47.9, 6.8]]
    >>> c = ChiSquareContingency(observed, expected)
    >>> c.test_summary
    {'association measures': {'C': 0.38790213046235816,
     'Cramers V': 0.2975893000268341,
     'phi-coefficient': 0.4208548241150648},
     'chi-square': 69.07632536255964,
     'continuity': True,
     'degrees of freedom': 6,
     'p-value': 6.323684774702373e-13}
    >>> c2 = ChiSquareContingency(observed)
    >>> c2.test_summary
    {'association measures': {'C': 0.3886475108354606,
     'Cramers V': 0.29826276547053077,
     'phi-coefficient': 0.4218072480793303},
     'chi-square': 69.3893282675805,
     'continuity': True,
     'degrees of freedom': 6,
     'p-value': 5.455268702303084e-13}
    >>> c2.expected
    array([[ 7.26923077, 30.32307692, 38.00769231,  5.4       ],
           [18.57692308, 77.49230769, 97.13076923, 13.8       ],
           [ 9.15384615, 38.18461538, 47.86153846,  6.8       ]])

    Notes
    -----
    The chi-square test is often used to assess the significance (if any) of the differences among :math:`k` different
    groups. The null hypothesis of the test, :math:`H_0` is typically that there is no significant difference between
    two or more groups.

    The chi-square test statistic, denoted :math:`\chi^2`, is defined as the following:

    .. math::

        \chi^2 = \sum^r_{i=1} \sum^k_{j=1} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}

    Where :math:`O_{ij}` is the ith observed frequency in the jth group and :math:`E_{ij}` is the corresponding
    expected frequency. The degrees of freedom is calculated as :math:`dof = (k - 1)(r - 1)` where :math:`k` is
    the number of columns and :math:`r` is the number of rows in the contingency table. In the case of a 2x2
    contingency table, Yates's continuity correction may be applied to reduce the error in approximation of using
    the chi-square distribution to calculate the test statistics. The continuity correction changes the
    computation of :math:`\chi^2` to the following:

    .. math::

        \chi^2 = \sum^r_{i=1} \sum^k_{j=1} \frac{(|O_{ij} - E_{ij}| - 0.5)^2}{E_{ij}}

    In addition to the test statistics, several measures of association are also provided. The first is the
    phi coefficient, defined as:

    .. math::

        \phi = \pm \sqrt{\frac{\chi^2}{N}}

    The contingency coefficient, denoted as :math:`C`, is defined as:

    .. math::

        C = \sqrt{\frac{\chi^2}{N + \chi^2}}

    Lastly, Cramer's V is defined as:

    .. math::

        V = \sqrt{\frac{\chi^2}{N(k-1}}

    References
    ----------
    Gibbons, J. D., & Chakraborti, S. (2010). Nonparametric statistical inference. London: Chapman & Hall.

    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2018, August 15). Contingency table. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:08, August 28, 2018,
        from https://en.wikipedia.org/w/index.php?title=Contingency_table&oldid=854973657

    Wikipedia contributors. (2017, October 20). Yates's correction for continuity. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:23, September 1, 2018,
        from https://en.wikipedia.org/w/index.php?title=Yates%27s_correction_for_continuity&oldid=806197753

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
        self.association_measures = self._assoc_measure()
        self.test_summary = {
            'chi-square': self.chi_square,
            'p-value': self.p_value,
            'association measures': self.association_measures,
            'degrees of freedom': self.degrees_freedom,
            'continuity': self.continuity
        }

    def _chi_square(self):
        r"""
        Computes the chi-square test statistic of the contingency table.

        Returns
        -------
        chi_val : float
            The computed chi-square test statistic.

        """
        cont_table = np.absolute(self.observed - self.expected)

        if self.degrees_freedom == 1:
            chi_val = np.sum((cont_table - (0.5 * self.continuity)) ** 2 / self.expected)
        else:
            chi_val = np.sum(cont_table ** 2 / self.expected)

        return chi_val

    def _p_value(self):
        r"""
        Calculates the p-value given the chi-square test statistic and the degrees of freedom.

        Returns
        -------
        pval : float
            The computed p-value.

        """
        pval = chi2.sf(self.chi_square, self.degrees_freedom)

        return pval

    def _assoc_measure(self):
        r"""
        Computes several contingency table association measures.

        Returns
        -------
        assocation_measures : dict
            A dictionary containing the calculated association measures, including the phi coefficient, C, and
            Cramer's V.

        """
        n = np.sum(self.observed)

        filled_diag = self.observed.copy()
        np.fill_diagonal(filled_diag, 1)

        phi_sign = np.prod(np.diagonal(self.observed)) - np.prod(filled_diag)

        phi_coeff = np.sqrt(self.chi_square / n)
        if phi_sign < 0 and phi_coeff > 0:
            phi_coeff = -phi_coeff

        c = np.sqrt(self.chi_square / (n + self.chi_square))

        v = np.sqrt(self.chi_square / (n * (np.minimum(self.observed.shape[0], self.observed.shape[1]) - 1)))

        association_measures = {
            'phi-coefficient': phi_coeff,
            'C': c,
            'Cramers V': v
        }

        return association_measures


class CochranQ(object):
    r"""
    Performs Cochran's Q test

    Parameters
    ----------
    sample1, sample2, ... : array-like
        One-dimensional array-like objects (numpy array, list, pandas DataFrame or pandas Series) containing the
        observed sample data. Each sample must be of the same length.

    Attributes
    ----------
    degrees_freedom : int
        Degrees of freedom is calculated as the number of samples minus 1.
    q_statistic : float
        The calculated Q test statistic.
    p_value : float
        The p-value of the test statistic.
    test_summary : dict
        Dictionary containing the relevant test results.

    Examples
    --------
    >>> r1 = [0,1,0,0,1,1,1,0,1,0,1,1,1,1,1,1,1,1]
    >>> r2 = [0,1,1,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1]
    >>> r3 = [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0]
    >>> cq = CochranQ(r1, r2, r3)
    >>> cq.test_summary
    {'degrees of freedom': 2,
     'p-value': 0.00024036947641951404,
     'q-statistic': 16.666666666666668}

    Notes
    -----
    Cochran's Q test is an extension of McNemar's test for two-way randomized block design experiments in which the
    response variable is binary (can only take one of two possible outcomes).

    Cochran's Q test is performed by arranging the group sample observation vectors into a two-way table consisting
    of :math:`n` rows and :math:`k` columns, where the binary responses are tallied as 1s ("successes") and 0s
    ("failures"). The :math:`Q` test statistic can then be calculated per the following definition:

    .. math::

        Q = \frac{(k - 1) \bigg[k \sum^k_{j=1} G_j^2 - \Big(\sum^k_{j=1} G_j \Big)^2 \bigg]}{k \sum^n_{i=1} L_i - \sum^n_{i=1} L_i^2

    Where :math:`G_j` is the sum of 1s ("successess") in the jth sample and :math:`L_i` is the sum of 1s ("successes")
    in the ith row.

    The distribution of :math:`Q` is approximated by the chi-square distribution with :math:`df = k - 1`.

    References
    ----------
    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2017, August 8). Cochran's Q test. In Wikipedia, The Free Encyclopedia.
        Retrieved 15:05, August 26, 2018,
        from https://en.wikipedia.org/w/index.php?title=Cochran%27s_Q_test&oldid=794571272

    """
    def __init__(self, *args):
        self.q_statistic, self.degrees_freedom = self._q_test(*args)
        self.p_value = self._p_value()
        self.test_summary = {
            'q-statistic': self.q_statistic,
            'p-value': self.p_value,
            'degrees of freedom': self.degrees_freedom,
        }

    @staticmethod
    def _q_test(*args):
        r"""

        Parameters
        ----------
        sample1, sample2, ... : array-like
            One-dimensional array-like objects (numpy array, list, pandas DataFrame or pandas Series) containing the
            observed sample data. Each sample must be of the same length.

        Returns
        -------
        tuple
            Tuple containing the computed Q test statistic and the degrees of freedom.

        """
        design_matrix = build_des_mat(*args, group=None)
        sample_counts = npi.group_by(design_matrix[:, 0], design_matrix[:, 1], np.sum)
        sample_size = design_matrix.shape[0] / len(np.unique(design_matrix[:, 0]))

        summary_table = pd.DataFrame(sample_counts, columns=['sample', '1s'])
        summary_table['sample_size'] = sample_size
        summary_table['0s'] = summary_table['sample_size'] - summary_table['1s']

        li2 = np.sum(np.sum(np.vstack([args]), axis=0) ** 2)

        k = summary_table.shape[0]
        dof = k - 1

        q = (dof *
             (k * np.sum(summary_table['1s'] ** 2) - np.sum(summary_table['1s']) ** 2)) / \
            (k * np.sum(summary_table['1s']) - li2)

        return q, dof

    def _p_value(self):
        pval = chi2.sf(self.q_statistic, self.degrees_freedom)

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
        self.test_summary = {
            'p-value': self.p_value,
            'odds ratio': self.odds_ratio,
            'contigency table': self.table
        }

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


class McNemarTest(object):
    r"""
    Computes the McNemar Test for two related samples in a 2x2 contingency table.

    Parameters
    ----------
    table : array-like
    continuity : bool, False

    Attributes
    ----------
    table : array-like
    continuity : bool
    n : int
    mcnemar_x2_statistic : float
    z_asymptotic_statistic : float
    mcnemar_p_value : float
    exact_p_value : float
    mid_p_value : float
    test_summary : dict

    Raises
    ------
    ValueError
        raised if the passed table has more than 2 columns or rows.
    ValueError
        raised if the table contains negative values.

    Notes
    -----

    Examples
    --------
    >>> m = McNemarTest([[59, 6], [16, 80]])
    >>> m.test_summary
    {'Asymptotic z-statistic': 2.1320071635561044,
     'Exact p-value': 0.052478790283203125,
     'McNemar p-value': 0.03300625766123255,
     'McNemar x2-statistic': 4.545454545454546,
     'Mid p-value': 0.034689664840698256,
     'N': 161,
     'continuity': False}
    >>> m2 = McNemarTest([[59, 6], [16, 80]], continuity=True)
    >>> m2.test_summary
    {'Asymptotic z-statistic': 1.9188064472004938,
     'Exact p-value': 0.052478790283203125,
     'McNemar p-value': 0.055008833629265896,
     'McNemar x2-statistic': 3.6818181818181817,
     'Mid p-value': 0.034689664840698256,
     'N': 161,
     'continuity': True}

    References
    ----------
    Fagerland, M. W., Lydersen, S., & Laake, P. (2013).
        The McNemar test for binary matched-pairs data: Mid-p and asymptotic are better than exact conditional.
        Retrieved April 14, 2018, from http://www.biomedcentral.com/1471-2288/13/91

    Gibbons, J. D., & Chakraborti, S. (2010). Nonparametric statistical inference. London: Chapman & Hall.

    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2018, April 29). McNemar's test. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:24, August 15, 2018,
        from https://en.wikipedia.org/w/index.php?title=McNemar%27s_test&oldid=838855782

    """
    def __init__(self, table, continuity=False):
        if not isinstance(table, np.ndarray):
            self.table = np.array(table)
        else:
            self.table = table

        if self.table.shape != (2, 2):
            raise ValueError("McNemar's Test requires a 2x2 contingency table with non-negative integers.")

        if (self.table < 0).any():
            raise ValueError('All values in table should be non-negative.')

        self.n = np.sum(self.table)
        self.continuity = continuity

        self.mcnemar_x2_statistic = self._mcnemar_test_stat()
        self.z_asymptotic_statistic = self._asymptotic_test()
        self.mcnemar_p_value = self._mcnemar_p_value()
        self.exact_p_value = self._exact_p_value()
        self.mid_p_value = self._mid_p_value()
        self.test_summary = {
            'N': self.n,
            'continuity': self.continuity,
            'McNemar x2-statistic': self.mcnemar_x2_statistic,
            'Asymptotic z-statistic': self.z_asymptotic_statistic,
            'McNemar p-value': self.mcnemar_p_value,
            'Exact p-value': self.exact_p_value,
            'Mid p-value': self.mid_p_value
        }

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

        return p_value * 2

    def _mid_p_value(self):
        n = self.table[1, 0] + self.table[0, 1]
        mid_p = self.exact_p_value - binom.pmf(self.table[0, 1], n, 0.5)

        return mid_p


def table_margins(table):
    r"""
    Computes the marginal sums of a one or two-dimensional array.

    Parameters
    ----------
    table : array-like

    Raises
    ------
    ValueError
        The given array must be either a one or two-dimensional array.

    Returns
    -------
    r, c : tuple

    Notes
    -----

    Examples
    --------

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
    r"""
    Computes the expected frequencies of a given contingency table with observed frequencies.

    Parameters
    ----------
    observed : array-like

    Raises
    ------
    ValueError
        the dimension of the :code:`observed` parameter cannot be greater than two.

    Returns
    -------
    exp_freq : array-like

    Examples
    --------

    Notes
    -----

    """
    if not isinstance(observed, np.ndarray):
        observed = np.array(observed).copy()

    if observed.ndim > 2:
        raise ValueError('table dimension cannot be greater than two.')

    margins = table_margins(observed)

    exp_freq = reduce(np.multiply, margins) / np.sum(observed)

    return exp_freq
