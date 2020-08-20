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

    table_margins
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
from itertools import combinations
import numpy as np
import numpy_indexed as npi
import pandas as pd
from scipy.special import comb
from scipy.stats import chi2, binom

from hypothetical._lib import _build_des_mat


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
    n : int
        Total number of samples
    chi_square : float
        The calculated chi-square value.
    p_value : float
        The associated p-value.
    cramers_v : float
        Cramer's V measure of association (dependence) inherent in the data contingency table.
    contingency_coefficient : float
        Contingency coefficient measure of association.
    phi_coefficient : float
        Phi coefficient of association in the data.
    tschuprows_coefficient : float
        Tschuprows coefficient for measure of association in the data.
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

    Cramer's V is defined as:

    .. math::

        V = \sqrt{\frac{\chi^2}{N(k-1}}

    Lastly, Tschuprow's T is defined as:

    .. math::

        T = \sqrt{\frac{\phi^2}{\sqrt{(r - 1)(c - 1)}}} = \sqrt{\frac{\frac{\chi^2}{n}}{\sqrt{(r - 1)(c - 1)}}}

    References
    ----------
    Gibbons, J. D., & Chakraborti, S. (2010). Nonparametric statistical inference. London: Chapman & Hall.

    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2018, August 15). Contingency table. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:08, August 28, 2018,
        from https://en.wikipedia.org/w/index.php?title=Contingency_table&oldid=854973657

    Wikipedia contributors. (2020, April 14). Cramér's V. In Wikipedia, The Free Encyclopedia.
        Retrieved 13:41, August 12, 2020,
        from https://en.wikipedia.org/w/index.php?title=Cram%C3%A9r%27s_V&oldid=950837942

    Wikipedia contributors. (2020, August 9). Phi coefficient. In Wikipedia, The Free Encyclopedia.
        Retrieved 13:40, August 12, 2020,
        from https://en.wikipedia.org/w/index.php?title=Phi_coefficient&oldid=971906217

    Wikipedia contributors. (2019, January 14). Tschuprow's T. In Wikipedia, The Free Encyclopedia.
        Retrieved 13:40, August 12, 2020,
        from https://en.wikipedia.org/w/index.php?title=Tschuprow%27s_T&oldid=878279875

    Wikipedia contributors. (2017, October 20). Yates's correction for continuity. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:23, September 1, 2018,
        from https://en.wikipedia.org/w/index.php?title=Yates%27s_correction_for_continuity&oldid=806197753

    https://www.empirical-methods.hslu.ch/decisiontree/relationship/chi-square-contingency/

    http://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf

    http://uregina.ca/~gingrich/ch11a.pdf

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
        self.n = np.sum(self.observed)
        self.chi_square = self._chi_square()
        self.p_value = self._p_value()
        self.cramers_v = self._cramers_v()
        self.contingency_coefficient = self._cont_coeff()
        self.phi_coefficient = self._phi_coeff()
        self.tschuprows_coefficient = self._tschuprows_coeff()
        self.test_summary = {
            'chi-square': self.chi_square,
            'p-value': self.p_value,
            "Cramer's V": self.cramers_v,
            'Contingency Coefficient': self.contingency_coefficient,
            'Phi Coefficient': self.phi_coefficient,
            'Tschuprow Coefficient': self._tschuprows_coeff(),
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

    def _cramers_v(self):
        r"""
        Computes Cramer's V measure of association between two data variables.

        Returns
        -------
        v : float

        """
        v = np.sqrt(self.chi_square / (self.n * (np.minimum(self.observed.shape[0], self.observed.shape[1]) - 1)))

        return v

    def _phi_coeff(self):
        r"""
        Computes the Phi (:math:`\phi`) coefficient measure of association.

        Returns
        -------
        phi_coeff : float

        """
        filled_diag = self.observed.copy()
        np.fill_diagonal(filled_diag, 1)

        phi_sign = np.prod(np.diagonal(self.observed)) - np.prod(filled_diag)

        phi_coeff = np.sqrt(self.chi_square / self.n)
        if phi_sign < 0 and phi_coeff > 0:
            phi_coeff = -phi_coeff

        return phi_coeff

    def _cont_coeff(self):
        r"""
        Returns the contingency coefficient :math:`C` measure of association.

        Returns
        -------
        c : float

        """
        c = np.sqrt(self.chi_square / (self.n + self.chi_square))

        return c

    def _tschuprows_coeff(self):
        r"""
        Returns Tschuprow's :math:`T` measure of association.

        Returns
        -------
        t : float

        """
        t = np.sqrt(self.chi_square
                    / (self.n * np.sqrt((self.observed.shape[0] - 1) * (self.observed.shape[1]) - 1)))

        return t


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

    https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Cochrans_Q_Test.pdf

    Wikipedia contributors. (2017, August 8). Cochran's Q test. In Wikipedia, The Free Encyclopedia.
        Retrieved 15:05, August 26, 2018,
        from https://en.wikipedia.org/w/index.php?title=Cochran%27s_Q_test&oldid=794571272
        
    """
    def __init__(self, *args, posthoc=False, names=None):
        self.design_matrix = _build_des_mat(*args, group=None)
        self.q_statistic, self.degrees_freedom = self._q_test(*args)
        self.p_value = self._p_value()

        self.test_summary = {
            'q-statistic': self.q_statistic,
            'p-value': self.p_value,
            'degrees of freedom': self.degrees_freedom,
        }

        if posthoc:
            self.posthoc = self._multiple_comparisons(*args, names=names)

        else:
            self.posthoc = 'None'

    def _q_test(self, *args):
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
        sample_lengths = []
        for i in args:
            sample_lengths.append(len(i))

        if len(set(sample_lengths)) != 1:
            raise ValueError('all sample observation vectors must have the same length.')

        sample_counts = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], np.sum)
        sample_size = self.design_matrix.shape[0] / len(np.unique(self.design_matrix[:, 0]))

        summary_table = pd.DataFrame(sample_counts, columns=['sample', '1s'])
        summary_table['0s'] = sample_size - summary_table['1s']

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

    def _multiple_comparisons(self, *args, names=None):
        if names is not None:
            if len(names) != len(np.unique(self.design_matrix[:, 0])):
                raise ValueError('group names array must be the same length as the number of sample groups.')

        else:
            names = np.unique(self.design_matrix[:, 0] + 1)

        dat = dict(zip(names, args))

        combs = [{j: dat[j] for j in i} for i in combinations(dat, 2)]

        group_comb = []
        q_stat = []
        p_val = []

        for comb in combs:
            name1, group1 = list(comb.keys())[0], list(comb.values())[0]
            name2, group2 = list(comb.keys())[1], list(comb.values())[1]

            c = CochranQ(group1, group2, names=[name1, name2])

            group_comb.append(str(name1) + ' : ' + str(name2))
            q_stat.append(c.q_statistic)
            p_val.append(c.p_value)

        result_df = pd.DataFrame({'groups': group_comb,
                                  'Q statistic': q_stat,
                                  'p-value': p_val})

        return result_df


class McNemarTest(object):
    r"""
    Computes the McNemar Test for two related samples in a 2x2 contingency table.

    Parameters
    ----------
    table : array-like
        Array-like object representing a 2x2 paired contingency table.
    continuity : bool, True
        Use continuity-corrected version of McNemar's chi-square test statistic as proposed by Edwards. Defaults to
        False as simulations performed by Fagerland (et al.) have shown the continuity-corrected version of
        McNemar's test to be overly conservative compared to the original McNemar test statistic.

    Attributes
    ----------
    table : array-like
        Array-like object representing a 2x2 paired contingency table.
    continuity : bool
        Apply continuity-corrected version of McNemar's chi-square statistic.
    n : int
        Total number of samples.
    mcnemar_x2_statistic : float
        The McNemar chi-square test statistic. If the parameter continuity is True, this value will be the
        continuity corrected version of the test statistic.
    z_asymptotic_statistic : float
        The test statistic of the asymptotic McNemar test. If the continuity parameter is True, the continuity
        corrected version of the asymptotic McNemar test as proposed by Edwards will be performed.
    mcnemar_p_value : float
        The p-value of the McNemar test statistic.
    exact_p_value : float
        The exact p-value of the McNemar test. The exact p-value is generally more accurate when the sample sizes
        of the data is small.
    mid_p_value : float
        The mid p-value of the McNemar test.
    test_summary : dict
        Dictionary containing relevant returned test statistics and entered parameters.

    Raises
    ------
    ValueError
        raised if the passed table has more than 2 columns or rows.
    ValueError
        raised if the table contains negative values.
    ValueError
        raised when table cell n_12 and n_21 are both 0.

    Notes
    -----
    McNemar's test is a test for paired data as in the case of 2x2 contingency tables with a dichotomous trait. The
    McNemar test determines if the row and column marginal frequencies are equal, which is also known as marginal
    homogeneity. For example, McNemar's test can be used when comparing postive/negative results for two tests,
    surgery vs. non-surgery in siblings and non-siblings, and other instances. The test was developed by Quinn
    McNemar in 1947.

    Consider a 2x2 contingency table with four cells where each cell and its position is denoted :math:`n_{rc}`
    where :math:`r = row` and :math:`c = column`. The appropriate null hypothesis states the marginal probabilities
    of each outcome are the same.

    .. math::

        n_{11} + n_{12} = n_{11} + n_{21}
        n_{12} + n_{22} = n_{21} + n_{22}

    The above simplifies to :math:`n_{12} = n_{21}`. Therefore the null hypothesis can be stated more simply as:

    .. math::

        H_0: n_{12} = n_{21}

    The null hypothesis can also be stated as the off-diagonal probabilities of the 2x2 contingency table are the
    same, with the alternative hypothesis stating the probabilities are not equal. To test this hypothesis, the
    McNemar test can be used, which is defined as:

    .. math::

        \chi^2 = \frac{(n_{12} - n_{21})^2}{n_{12} + n_{21}}

    This is also known as the asymptotic McNemar test. With an adequate number of samples, the McNemar test
    statistic, :math:`\chi^2` has a chi-square distribution with one degree of freedom.

    Continuity correction can be applied to the asymptotic McNemar test as proposed by Edwards [1]. The continuity
    corrected version of the asymptotic McNemar test approximates the McNemar exact conditional test which is
    described below. The asymptotic McNemar test with continuity correction is defined as:

    .. math::

        z = \frac{|n_{12} - n_{21}| - 1}{\sqrt{n_{12} + n_{21}}}

    Fagerland et al [1] recommend the asymptotic McNemar test in most cases. The continuity corrected version is
    not recommended as it has been shown to be overly conservative.

    There also exists several variations of the original McNemar test that may have better performance in specific
    cases.

    Variations of the McNemar Test

    When the sample sizes of cells :math:`n_{12}` or `n_{21}` are small (small being subjective, but generally
    assumed to be < 30), an exact binomial test can be used to calculate McNemar's test. This is known as the
    McNemar exact conditional test. The one-sided test is defined as the following:

    .. math::

        p_{exact} = \sum^n_{i=n_{12}} \binom{n}{i} \frac{1}{2}^i (1 - \frac{1}{2})^{n - i}

    The two-sided p-value can also be easily found by multiplying :math:`p_{exact}` by :math:`2`.

    Fagerland et al[2] do not recommend the exact conditional test as it was found to have least the performance
    Type 1 error and power of other McNemar test variations.

    The McNemar mid-p test is calculated by subtracting half the point probability of the observed :math:`n_{12}`
    cell of the contingency table from the one-sided :math:`p_{exact}` value using the equation above. The
    resulting p-value is then doubled to obtain the two-sided mid-p-value. Stated more formally, the McNemar
    mid-p test is defined as:

    .. math::

        p_{mid} = 2 \large(\sum^n_{i=b} \binom{n}{i} \frac{1}{2}^i (1 - \frac{1}{2})^{n - i} -
        \frac{1}{2} \binom{n}{b} \frac{1}{2}^b (1 - \frac{1}{2}^{n -b } \large)

    According to Fagerland et al [2], the McNemar mid-p test has much higher performance compared to the
    McNemar exact conditional test and is considerable alternative to the McNemar exact unconditional test which
    is significantly more complex.

    Examples
    --------
    >>> m = McNemarTest([[59, 6], [16, 80]], continuity=False)
    >>> m.test_summary
    {'Asymptotic z-statistic': 2.1320071635561044,
     'Exact p-value': 0.052478790283203125,
     'McNemar p-value': 0.03300625766123255,
     'McNemar x2-statistic': 4.545454545454546,
     'Mid p-value': 0.034689664840698256,
     'N': 161,
     'continuity': False}
    >>> m2 = McNemarTest([[59, 6], [16, 80]])
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
    Edwards AL: Note on the “correction for continuity” in testing the
        significance of the difference between correlated proportions.
        Psychometrika 1948, 13(3):185–187.

    Fagerland, M. W., Lydersen, S., & Laake, P. (2013).
        The McNemar test for binary matched-pairs data: Mid-p and asymptotic are better than exact conditional.
        Retrieved April 14, 2018, from http://www.biomedcentral.com/1471-2288/13/91

    Gibbons, J. D., & Chakraborti, S. (2010). Nonparametric statistical inference. London: Chapman & Hall.

    Wikipedia contributors. (2018, April 29). McNemar's test. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:24, August 15, 2018,
        from https://en.wikipedia.org/w/index.php?title=McNemar%27s_test&oldid=838855782

    """
    def __init__(self, table, continuity=True):

        if not isinstance(table, np.ndarray):
            self.table = np.array(table)
        else:
            self.table = table

        if self.table.shape != (2, 2):
            raise ValueError("McNemar's Test requires a 2x2 contingency table with non-negative integers.")

        if (self.table < 0).any():
            raise ValueError('All values in table should be non-negative.')

        if self.table[0, 1] == 0 and self.table[1, 0] == 0:
            raise ValueError('Entered values n_12 and n_21 cannot be zero.')

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
        r"""
        Calculates the McNemar test statistic.

        Returns
        -------
        x2 : float
            The computed McNemar test statistic.

        Notes
        -----
        The McNemar test statistic is calculated as:

        .. math::

            \chi^2 = \frac{(n_{12} - n_{21})^2}{n_{12} + n_{21}}

        """
        if not self.continuity:
            x2 = (self.table[0, 1] - self.table[1, 0]) ** 2 / (self.table[0, 1] + self.table[1, 0])
        else:
            x2 = (np.absolute(self.table[0, 1] - self.table[1, 0]) - 1) ** 2 / (self.table[0, 1] + self.table[1, 0])

        return x2

    def _mcnemar_p_value(self):
        r"""
        Computes the p-value of the asymptotic McNemar test.

        Returns
        -------
        p : float
            The p-value of the :math:`\chi^2` statistic.

        Notes
        -----
        The McNemar test statistic has a chi-square distribution.

        """
        p = 1 - chi2.cdf(self.mcnemar_x2_statistic, 1)

        return p

    def _asymptotic_test(self):
        r"""
        Calculates the asymptotic McNemar test.

        Returns
        -------
        z_asymptotic : float
            The computed asymptotic McNemar test statistic.

        """
        if not self.continuity:
            z_asymptotic = (self.table[0, 1] - self.table[1, 0]) / np.sqrt(self.table[0, 1] + self.table[1, 0])
        else:
            z_asymptotic = (np.absolute(self.table[0, 1] - self.table[1, 0]) - 1) / \
                           np.sqrt(self.table[0, 1] + self.table[1, 0])

        return z_asymptotic

    def _exact_p_value(self):
        r"""
        Computes the exact p-value of the McNemar test.

        Returns
        -------
        p_value : float
            The calculated exact p-value.

        Notes
        -----
        The one-sided exact p-value is defined as the following:

        .. math::

            p_{exact} = \sum^n_{i=n_{12}} \binom{n}{i} \frac{1}{2}^i (1 - \frac{1}{2})^{n - i}

        """
        i = self.table[0, 1]
        n = self.table[1, 0] + self.table[0, 1]
        i_n = np.arange(i + 1, n + 1)

        p_value = 1 - np.sum(comb(n, i_n) * 0.5 ** i_n * (1 - 0.5) ** (n - i_n))

        return p_value * 2

    def _mid_p_value(self):
        r"""
        Calculates the mid-p-value of the McNemar test.

        Returns
        -------
        mid_p : float
            The mid-p value.

        Notes
        -----
        The McNemar mid-p test is calculated by subtracting half the point probability of the observed :math:`n_{12}`
        cell of the contingency table from the one-sided :math:`p_{exact}` value using the equation above. The
        resulting p-value is then doubled to obtain the two-sided mid-p-value. Stated more formally, the McNemar
        mid-p test is defined as:

        .. math::

            p_{mid} = 2 \large(\sum^n_{i=b} \binom{n}{i} \frac{1}{2}^i (1 - \frac{1}{2})^{n - i} -
            \frac{1}{2} \binom{n}{b} \frac{1}{2}^b (1 - \frac{1}{2}^{n -b } \large)

        """
        n = self.table[1, 0] + self.table[0, 1]
        mid_p = self.exact_p_value - binom.pmf(self.table[0, 1], n, 0.5)

        return mid_p


def table_margins(table):
    r"""
    Computes the marginal sums of a given array.

    Parameters
    ----------
    table : array-like
        A one or two-dimensional array-like object.

    Raises
    ------
    ValueError
        The given array must be either a one or two-dimensional array.

    Returns
    -------
    r, c : tuple
        A tuple containing the total sums of the table rows and the total sums of the table columns.

    Examples
    --------
    >>> t = table_margins([[10, 10, 20], [20, 20, 10]])

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
        A one or two-dimensional array object.

    Raises
    ------
    ValueError
        the dimension of the :code:`observed` parameter cannot be greater than two.

    Returns
    -------
    exp_freq : array-like
        Array of the expected frequencies

    Examples
    --------
    >>> expected_frequencies([[10, 10, 20], [20, 20, 10]])
    array([[13.33333333, 13.33333333, 13.33333333],
           [16.66666667, 16.66666667, 16.66666667]])

    Notes
    -----
    The expected frequency, here denoted as :math:`E_{cr}`, where :math:`c` is the column index and :math:`r` is the
    row index. Stated more formally, the expected frequency is defined as:

    .. math::

        E_{cr} = \frac{(\sum^{n_r}_{i=0} r_i)(\sum^{n_c}_{i=0} c_i)}{n}

    Where :math:`n` is the total sample size and :math:`n_c, n_r` are the number of cells in row and column,
    respectively. The expected frequency is calculated for each 'cell' in the given array.

    References
    ----------
    Stover, Christopher. "Contingency Table."
        From MathWorld--A Wolfram Web Resource, created by Eric W. Weisstein.
        http://mathworld.wolfram.com/ContingencyTable.html

    """
    if not isinstance(observed, np.ndarray):
        observed = np.array(observed).copy()

    if observed.ndim > 2:
        raise ValueError('table dimension cannot be greater than two.')

    margins = table_margins(observed)

    exp_freq = reduce(np.multiply, margins) / np.sum(observed)

    return exp_freq


# class FisherTest(object):
#     r"""
#     Performs Fisher's Exact Test for a 2x2 contingency table.
#
#     Parameters
#     ----------
#
#     Attributes
#     ----------
#
#     Notes
#     -----
#
#     Examples
#     --------
#
#     References
#     ----------
#     Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
#         McGraw-Hill. ISBN 07-057348-4
#
#     Weisstein, Eric W. "Fisher's Exact Test." From MathWorld--A Wolfram Web Resource.
#         http://mathworld.wolfram.com/FishersExactTest.html
#
#     Wikipedia contributors. (2018, May 20). Fisher's exact test. In Wikipedia, The Free Encyclopedia.
#         Retrieved 12:46, August 14, 2018,
#         from https://en.wikipedia.org/w/index.php?title=Fisher%27s_exact_test&oldid=842100719
#
#     """
#     def __init__(self, table=None, alternative='two-sided'):
#         if not isinstance(table, np.ndarray):
#             self.table = np.array(table)
#         else:
#             self.table = table
#
#         if self.table.shape != (2, 2):
#             raise ValueError("Fisher's Exact Test requires a 2x2 contingency table with non-negative integers.")
#
#         if (self.table < 0).any():
#             raise ValueError('All values in table should be non-negative.')
#
#         self.n = np.sum(self.table)
#         self.p_value = self._p_value()
#         self.odds_ratio = self._odds_ratio()
#         self.test_summary = {
#             'p-value': self.p_value,
#             'odds ratio': self.odds_ratio,
#             'contigency table': self.table
#         }
#
#     def _p_value(self):
#         a, b, c, d = self.table[0, 0], self.table[0, 1], self.table[1, 0], self.table[1, 1]
#
#         p = (comb(a + c, a) * comb(b + d, b)) / comb(self.n, a + b)
#
#         return p
#
#     def _odds_ratio(self):
#         if self.table[1, 0] > 0 and self.table[0, 1] > 0:
#             oddsratio = self.table[0, 0] * self.table[1, 1] / (self.table[1, 0] * self.table[0, 1])
#         else:
#             oddsratio = np.inf
#
#         return oddsratio
