# encoding=utf8


"""
Functions for performing nonparametric statistical inference.

Nonparametric Inference Methods
-------------------------------

.. autosummary::
    :toctree: generated/

    FriedmanTest
    KruskalWallis
    MannWhitney
    MedianTest
    RunsTest
    SignTest
    VanDerWaerden
    WaldWolfowitz
    WilcoxonTest

Other Functions
---------------

.. autosummary::
    :toctree: generated/

    tie_correction
    count_runs

References
----------
Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
    Wiley. ISBN 978-1118840313.

Fox J. and Weisberg, S. (2011) An R Companion to Applied Regression, Second Edition Sage.

Mann–Whitney U test. (2017, June 20). In Wikipedia, The Free Encyclopedia.
    From https://en.wikipedia.org/w/index.php?title=Mann%E2%80%93Whitney_U_test&oldid=786593885

Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
    McGraw-Hill. ISBN 07-057348-4

Wikipedia contributors. (2018, August 20). Friedman test. In Wikipedia, The Free Encyclopedia.
    Retrieved 12:56, August 27, 2018,
    from https://en.wikipedia.org/w/index.php?title=Friedman_test&oldid=855731754

Wikipedia contributors. (2018, May 21). Kruskal–Wallis one-way analysis of variance.
    In Wikipedia, The Free Encyclopedia. From
    https://en.wikipedia.org/w/index.php?title=Kruskal%E2%80%93Wallis_one-way_analysis_of_variance&oldid=842351945

Wikipedia contributors. (2017, June 27). Median test. In Wikipedia, The Free Encyclopedia.
    Retrieved 12:23, August 19, 2018, from https://en.wikipedia.org/w/index.php?title=Median_test&oldid=787822318

Wikipedia contributors. (2018, August 22). Wald–Wolfowitz runs test. In Wikipedia, The Free Encyclopedia.
        Retrieved 13:54, September 13, 2018,
        from https://en.wikipedia.org/w/index.php?title=Wald%E2%80%93Wolfowitz_runs_test&oldid=856082551

"""

from itertools import groupby, combinations
from collections import Counter
import numpy as np
import numpy_indexed as npi
import pandas as pd
from scipy.stats import chi2, norm, rankdata, t, find_repeats
from scipy.special import comb

from hypothetical._lib import _build_des_mat, _rank, _group_rank_sums
from hypothetical.descriptive import var
from hypothetical.hypothesis import BinomialTest
from hypothetical.contingency import ChiSquareContingency
from hypothetical.critical import r_critical_value


class FriedmanTest(object):
    r"""
    Performs the Friedman nonparametric test for multiple matched samples on an ordinal scale.

    Parameters
    ----------
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector.
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector.

    Attributes
    ----------
    design_matrix : array-like
        Numpy ndarray representing the data matrix for the analysis.
    n : int
        The number of samples in the design_matrix,
    k : int
        Number of groups in design matrix.
    xr2 : float
        The Friedman test statistic.
    p-value: float
        Associated p-value of the Friedman test statistic.
    summary : dict
        Dictionary containing test summary results including the :math:`xr2` and :math:`p-value`.

    Examples
    --------

    Notes
    -----
    The Friedman test casts the given data into a matrix of :math:`n` rows (number of samples in data) and :math:`k`
    columns (the number of sample groups). The data in each column is then ranked separately, meaning the range of
    any row of ranks will be between :math:`1` and :math:`k` is the number of groups, or 'treatments'. The Friedman
    test then determines whether the sample data is likely to have come from the same population.

    The test statistic of the Friedman test is :math:`\chi_r^2`. The test statistic's distribution resembles a
    chi-square distribution with degrees of freedom :math:`k - 1` when the samples and groups is sufficiently
    large ('sufficiently' being somewhat arbitrary).

    The Friedman test statistic :math:`\chi_r^2` is defined as:

    .. math::

         :math:`\chi_r^2` = \frac{12}{Nk(k+1)} \sum^k_{j=1} (R_j)^2 - 3N(k + 1)

    where :math:`N` is the number of rows (samples), :math:`k` is the number of columns (groups/treatments) and
    :math:`R_j` is the sum of the ranks in the :math:`j^{th} column.

    The Friedman test sometimes uses :math:`Q` as a test statistic with a slightly different definition:

    .. math::

        Q = \frac{12n}{k(k+1)} \sum^k_{j=1} (\bar{r}_j - \frac{k+1}{2})^2

    where :math:`\bar{r}_j` is the sum of the ranked data in the :math:`r^{th}` row.

    .. math::

        \bar{r}_j = \frac{1}{n} \sum^n_{i=1} r_{ij}

    When ties exist in the data, the :math:`Q` definition of the Friedman test statistic changes to:

    .. math::

        Q = \frac{(k-1) \sum^k_{i=1} (R_i - \frac{n(k+1){2})^2}{A_1 - C_1}

    where:

    .. math::

        A_1 = \sum^n_{i=1} \sum^k_{j=1} (R(X_{ij}))^2
        C_1 = \frac{nk(k+1)^2}{4}

    Another approach for correcting ties in the data is the following:

    .. math::

        Q_{adj} = frac{Q}{C}

    Where :math:`C` is a tie correction factor defined as:

    .. math::

        C = 1 - \frac{\sum (t^3 - t_i)}{n(k^3 - k)}

    Where :math:`t_i` is the number of tied scores in the :math:`i^{th}` set of ties.

    References
    ----------
    Gibbons, J. D., & Chakraborti, S. (2010). Nonparametric statistical inference. London: Chapman & Hall.

    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2018, August 20). Friedman test. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:56, August 27, 2018,
        from https://en.wikipedia.org/w/index.php?title=Friedman_test&oldid=855731754

    """
    def __init__(self, *args, group):
        if group is not None and len(args) > 1:
            raise ValueError('Only one sample vector should be passed when including a group vector')

        self.design_matrix = _build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.n, self.k = self.design_matrix.shape

        self.xr2 = self._xr2_test()
        self.p_value = self._p_value()
        self.test_summary = {
            'xr2 statistic': self.xr2,
            'p-value': self.p_value
        }

    def _xr2_test(self):
        r"""
        Computes the Friedman test statistic.

        Returns
        -------
        xr2 : float
            The Friedman test statistic.

        Notes
        -----
        The Friedman test statistic :math:`\chi_r^2` is defined as:

        .. math::

             :math:`\chi_r^2` = \frac{12}{Nk(k+1)} \sum^k_{j=1} (R_j)^2 - 3N(k + 1)

        where :math:`N` is the number of rows (samples), :math:`k` is the number of columns (groups/treatments) and
        :math:`R_j` is the sum of the ranks in the :math:`j^{th} column.

        The Friedman test sometimes uses :math:`Q` as a test statistic with a slightly different definition:

        .. math::

            Q = \frac{12n}{k(k+1)} \sum^k_{j=1} (\bar{r}_j - \frac{k+1}{2})^2

        where :math:`\bar{r}_j` is the sum of the ranked data in the :math:`r^{th}` row.

        .. math::

            \bar{r}_j = \frac{1}{n} \sum^n_{i=1} r_{ij}

        """
        ranks = []
        for i in range(self.n):
            ranks.append(rankdata(self.design_matrix[i]))

        ranks = np.vstack(ranks)

        ties = []

        for i in range(self.n):
            repeat_count = list(find_repeats(self.design_matrix[i])[1])
            if repeat_count:
                ties.append(repeat_count)

        correction = 1 - np.sum(np.array(ties) ** 3 - np.array(ties)) / (self.n * (self.k ** 3 - self.k))

        xr2 = (12. / (self.n * self.k * (self.k + 1.))) * np.sum(np.sum(ranks, axis=0) ** 2.) - (
                    3. * self.n * (self.k + 1.))

        xr2 /= correction

        return xr2

    def _p_value(self):
        r"""
        Returns the p-value of the Freidman test.

        Returns
        -------
        pval: float
            The p-value of the Friedman test statistic given a chi-square distribution.

        """
        pval = chi2.sf(self.xr2, self.k - 1)

        return pval


class KruskalWallis(object):
    r"""
    Class containing the algorithms and methods used in the construction and conduction of the
    Kruskal-Wallis H-test.

    Attributes
    ----------
    design_matrix : array-like
        Numpy ndarray representing the data matrix for the analysis.
    ranked_matrix : array-like
        Numpy ndarray representing the data matrix with the ranked observations.
    alpha : float
        Alpha level for determining significance.
    n : int
        Number of sample observations.
    k : int
        Number of treatment groups.
    dof : int
        Degrees of freedom, defined as :math:`k - 1`.
    H : float
        Calculated Kruskal-Wallis H-statistic.
    t_value : float
        The critical t-value for computing the Least Significant Difference.
    p_value : float
        Corresponding p-value of the :math:`H`-statistic. The distribution of :math:`H` is approximated
        by the chi-square distribution.
    least_significant_difference : float
        Calculated Least Significant Difference for determining if treatment group means are significantly
        different from each other.
    test_description : str
        String describing the performed test. By default, the test description will be Kruskal-Wallis rank sum test

    Parameters
    ----------
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector.
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector.
    alpha : float
        Desired alpha level for testing for significance.

    Raises
    ------
    ValueError
        As the Kruskal-Wallis is a univariate test, only one sample observation vector should be passed
        when including a group vector in the :code:`group` parameter.

    Notes
    -----
    The Kruskal-Wallis test extends the Mann-Whitney U test for more than two groups and can be
    considered the nonparametric equivalent of the one-way analysis of variance (ANOVA) method.
    The test is nonparametric similar to the Mann-Whitney test and as such does not
    assume the data are normally distributed and can, therefore, be used when the assumption
    of normality is violated.

    The Kruskal-Wallis test proceeds by ranking the data from 1 (the smallest) to the largest
    with ties replaced by the mean of the ranks the values would have received. The sum of
    the ranks for each treatment is typically denoted $T_i$ or $R_i$.

    The test statistic is denoted :code:`H` and can be defined as the following when the
    ranked data does not contain ties.

    .. math::

        H = \frac{12}{N(N + 1)} \left[ \frac{\sum_{i=1}^k T_{i}^2}{n_i} - 3(N + 1) \right]

    If the ranked data contains ties, a correction can be used by dividing :code:`H` by:

    .. math::

        1 - \frac{\sum_{t=1}^G (t_i^3 - t_i)}{N^3 - N}

    Where :code:`G` is the number of groups of tied ranks and :code:`t_i` is the number of
    tied values within the :code:`i^{th}` group. The p-value is usually approximated using
    a Chi-Square distribution as calculating exact probabilities can be computationally
    intensive for larger sample sizes.

    See Also
    --------
    AnovaOneWay : class containing the implementations of the algorithms and methods used in the
        conduction of the one-way analysis of variance procedure. The Kruskal-Wallis test can be
        considered the nonparametric equivalent of the one-way analysis of variance method.

    Examples
    --------
    There are several ways to perform the Kruskal-Wallis test with the :code:`kruskal_wallis` function.
    Similar to the parametric one-way ANOVA method implemented by the :code:`anova_one_way` function,
    one approach is to pass a group vector with the :code:`group` parameter and the corresponding
    observation vector as below.

    The data used in this example is a subset of the data obtained from the plant growth
    dataset given in Dobson (1983).

    >>> group_vector = ['ctrl', 'ctrl', 'ctrl',
    ...                 'trt1', 'trt1', 'trt1',
    ...                 'trt2', 'trt2', 'trt2']
    >>> observation_vec = [4.17, 5.58, 5.18,
    ...                    4.81, 4.17, 4.41,
    ...                    5.31, 5.12, 5.54]
    >>> kw = KruskalWallis(observation_vec, group=group_vector)
    >>> kw.test_summary
    {'alpha': 0.05,
     'critical chisq value': 3.1148459383753497,
     'degrees of freedom': 2,
     'least significant difference': 4.916428084371546,
     'p-value': 0.21067829669685478,
     't-value': 2.4469118487916806,
     'test description': 'Kruskal-Wallis rank sum test'}

    The other approach is to pass each group sample vector similar to the below.

    >>> ctrl = [4.17, 5.58, 5.18]
    >>> trt1 = [4.81, 4.17, 4.41]
    >>> trt2 = [5.31, 5.12, 5.54]
    >>> kw2 = KruskalWallis(ctrl, trt1, trt2)
    >>> kw2.test_summary
    {'alpha': 0.05,
     'critical chisq value': 3.1148459383753497,
     'degrees of freedom': 2,
     'least significant difference': 4.916428084371546,
     'p-value': 0.21067829669685478,
     't-value': 2.4469118487916806,
     'test description': 'Kruskal-Wallis rank sum test'}

    References
    ----------
    Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
        Wiley. ISBN 978-1118840313.

    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2018, May 21). Kruskal–Wallis one-way analysis of variance.
        In Wikipedia, The Free Encyclopedia. From
        https://en.wikipedia.org/w/index.php?title=Kruskal%E2%80%93Wallis_one-way_analysis_of_variance&oldid=842351945

    """
    def __init__(self, *args, group=None, alpha=0.05):

        if group is not None and len(args) > 1:
            raise ValueError('Only one sample vector should be passed when including a group vector')

        self.design_matrix = _build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.ranked_matrix = _rank(self.design_matrix)
        self.group_rank_sums = _group_rank_sums(self.ranked_matrix)
        self.alpha = alpha
        self.n = self.design_matrix.shape[0]
        self.k = len(np.unique(self.design_matrix[:, 0]))
        self.dof = self.k - 1
        self.H = self._h_statistic()
        self.p_value = self._p_value()
        self.t_value = self._t_value()
        self.least_significant_difference = self._lsd()
        self.test_description = 'Kruskal-Wallis rank sum test'
        self.test_summary = {'test description': self.test_description,
                             'critical chisq value': self.H,
                             'p-value': self.p_value,
                             'least significant difference': self.least_significant_difference,
                             't-value': self.t_value,
                             'alpha': self.alpha,
                             'degrees of freedom': self.dof
                             }

    def _h_statistic(self):
        r"""
        Computes the Kruskal-Wallis :math:`H`-statistic.

        Returns
        -------
        h : float
            Computed Kruskal-Wallis :math:`H`-statistic.

        Notes
        -----
        The Kruskal-Wallis :math:`H`-statistic is defined as the following when the ranked data does not
        contain ties.

        .. math::

            H = \frac{12}{N(N + 1)} \left[ \frac{\sum_{i=1}^k T_{i}^2}{n_i} - 3(N + 1) \right]

        If the ranked data contains ties, a correction can be used by dividing :code:`H` by:

        .. math::

            1 - \frac{\sum_{t=1}^G (t_i^3 - t_i)}{N^3 - N}

        Where :code:`G` is the number of groups of tied ranks and :code:`t_i` is the number of
        tied values within the :code:`i^{th}` group.

        The tie correction is automatically applied in the computation of the :math:`H`-statistic.

        References
        ----------
        Wikipedia contributors. (2018, May 21). Kruskal–Wallis one-way analysis of variance.
            In Wikipedia, The Free Encyclopedia. From
            https://en.wikipedia.org/w/index.php?title=Kruskal%E2%80%93Wallis_one-way_analysis_of_variance&oldid=842351945

        """
        group_observations = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1:], len)

        group_observations = np.array([i for _, i in group_observations])

        group_summed_ranks = np.array([i for _, i in self.group_rank_sums])

        h1 = 12. / (self.n * (self.n + 1))
        h2 = np.sum(group_summed_ranks ** 2 / group_observations)

        h = h1 * h2 - (3 * (self.n + 1))

        # Apply tie correction
        h /= tie_correction(self.ranked_matrix[:, 2])

        return h

    def _p_value(self):
        r"""
        Computes the p-value of the :math:`H`-statistic approximated by the chi-square distribution.

        Returns
        -------
        p : float
            The computed p-value.

        Notes
        -----
        The :math:`p`-value is approximated by a chi-square distribution with :math:`k - 1` degrees
        of freedom.

        .. math::

            Pr(\chi^2_{k - 1} \geq H)

        References
        ----------
        Wikipedia contributors. (2018, May 21). Kruskal–Wallis one-way analysis of variance.
            In Wikipedia, The Free Encyclopedia. From
            https://en.wikipedia.org/w/index.php?title=Kruskal%E2%80%93Wallis_one-way_analysis_of_variance&oldid=842351945

        """
        p = 1 - chi2.cdf(self.H, self.dof)

        return p

    def _t_value(self):
        r"""
        Returns the critical t-statistic given the input alpha-level (defaults to 0.05).

        Returns
        -------
        tval : float
            The critical t-value for using in computing the Least Significant Difference.

        Notes
        -----
        Scipy's :code:`t.ppf` method is used to compute the critical t-value.

        """
        tval = t.ppf(1 - self.alpha / 2, self.n - self.k)

        return tval

    def _lsd(self):
        r"""
        Returns the Least Significant Difference statistic used for determining if treatment group
        means are significantly different from each other.

        Returns
        -------
        lsd : float
            The calculated Least Significant Difference.

        Notes
        -----
        The Least Significant Difference is a test statistic developed by Ronald Fisher. The basic
        idea of the LSD is to find the smallest difference between two sample means and conclude a
        significant difference if a comparison between two other group means exceeds the LSD. The
        Least Significant Difference is defined as:

        .. math::

            t_{\alpha, N-k} \sqrt{MSE \frac{2}{n}}

        Where :math:`t_{\alpha, N-k}` is the critical t-value given the input alpha-level and :math:`MSE`
        is the mean error sum of squares as in the one-way analysis of variance procedure.

        References
        ----------
        Fisher’s Least Significant Difference (LSD) Test. (2010). [ebook] Thousand Oaks.
            Available at: https://www.utd.edu/~herve/abdi-LSD2010-pretty.pdf [Accessed 11 Jun. 2018].

        """
        lsd = self.t_value * np.sqrt(self._mse() * 2 / (self.n / self.k))

        return lsd

    def _rank(self):

        ranks = rankdata(self.design_matrix[:, 1], 'average')

        ranks = np.column_stack([self.design_matrix, ranks])

        return ranks

    def _group_rank_sums(self):
        rank_sums = npi.group_by(self.ranked_matrix[:, 0], self.ranked_matrix[:, 2], np.sum)

        return rank_sums

    def _mse(self):
        group_variance = npi.group_by(self.ranked_matrix[:, 0], self.ranked_matrix[:, 2], var)
        group_n = npi.group_by(self.ranked_matrix[:, 0], self.ranked_matrix[:, 2], len)

        sse = 0

        for i, j in zip(group_n, group_variance):
            sse += (i[1] - 1) * j[1]

        return sse / (self.n - self.k)


class MannWhitney(object):
    r"""
    Performs the nonparametric Mann-Whitney U test of two independent sample groups.

    Parameters
    ----------
    y1 : array-like
        One-dimensional array-like (Pandas Series or DataFrame, Numpy array, or list)
        designating first sample observation values.
    y2 : array-like, optional
        One-dimensional array-like (Pandas Series or DataFrame, Numpy array, or list)
        designating second sample observation values.
    group : array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series or DataFrame, or list) that defines
        the group membership of the sample vector(s). Must be the same length as the observation vector.
    continuity : bool
        If True, apply the continuity correction of :math:`\frac{1}{2}` to the
        mean rank.

    Attributes
    ----------
    y1 : array-like
        First sample observation vector.
    y2 : array-like or None
        Second sample observation vector, if passed. Otherwise, will return None.
    n1 : int
        Number of sample observations in the first sample vector.
    n2 : int or None
        Number of sample observations in the second sample vector. If no second observation vector was
        passed, will return None.
    n : int
        Total number of sample observations (sum of :code:`n1` and :code:`n2`.
    continuity : bool
        If True, continuity correction is applied.
    U : int
        Computed U-statistic.
    meanrank : float
        The mean of the ranked sample observations.
    sigma : float
        The calculated standard deviation, :math:`\sigma_U`.
    z_value : float
        Standardized :math:`z` value.
    p_value : float
        Computed p-value.
    effect_size : float
        Calculated estimated Cohen's effect size.

    Notes
    -----
    The Mann-Whitney U test is a nonparametric hypothesis test that tests the null hypothesis that
    there is an equally likely chance that a randomly selected observation from one sample will be
    less than or greater than a randomly selected observation from a second sample. Nonparametric
    methods are so named since they do not rely on the assumption of normality of the data.

    The test statistic in the Mann-Whitney setting is denoted as :math:`U` and is the minimum of
    the summed ranks of the two samples. The null hypothesis is rejected if :math:`U \leq U_0`,
    where :math:`U_0` is found in a table for small sample sizes. For larger sample sizes,
    :math:`U` is approximately normally distributed.

    The test is nonparametric in the sense it uses the ranks of the values rather than the values
    themselves. Therefore, the values are ordered then ranked from 1 (smallest value) to the largest
    value. Ranks of tied values get the mean of the ranks the values would have received. For example,
    for a set of data points :math:`\{4, 7, 7, 8\}` the ranks are :math:`\{1, 2.5, 2.5, 4\}`. The
    :math:`2.5` rank comes from :math:`2 + 3 = 5 / 2`. The ranks are then added for the values for
    both samples. The sum of the ranks for each sample are typically denoted by :math:`R_k` where
    :math:`k` is a sample indicator.

    :math:`U` for the two samples in the test, is given by:

    .. math::

        U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}
        U_2 = R_2 - \frac{n_2(n_2 + 1)}{2}

    Where :math:`R_1` and :math:`R_2` are the sum of the ranks of the two samples.

    Examples
    --------
    Similar to the :code:`anova_one_way` function, there are several ways to perform a Mann-Whitney
    U test with the :code:`mann_whitney` function. One of these approaches is to pass the sample data
    vector and a group vector of the same length denoting group membership of the sample observations.

    The data used in this example is a subset of the professor salary dataset found in Fox and
    Weisberg (2011).

    >>> professor_discipline = ['B', 'B', 'B', 'B', 'B',
    ...                         'A', 'A', 'A', 'A', 'A']
    >>> professor_salary = [139750, 173200, 79750, 11500, 141500,
    ...                     103450, 124750, 137000, 89565, 102580]
    >>> mw = MannWhitney(group=professor_discipline, y1=professor_salary)
    >>> mw.test_summary
    {'U': 10.0,
     'continuity': True,
     'mu meanrank': 13.0,
     'p-value': 0.5308693039685082,
     'sigma': 4.7871355387816905,
     'test description': 'Mann-Whitney U test',
     'z-value': 0.6266795614405122}

    The other approach is to pass each group sample observation vector.

    >>> sal_a = [139750, 173200, 79750, 11500, 141500]
    >>> sal_b = [103450, 124750, 137000, 89565, 102580]
    >>> mw2 = MannWhitney(sal_a, sal_b)
    >>> mw2.test_summary
    {'U': 10.0,
     'continuity': True,
     'mu meanrank': 13.0,
     'p-value': 0.5308693039685082,
     'sigma': 4.7871355387816905,
     'test description': 'Mann-Whitney U test',
     'z-value': 0.6266795614405122}

    References
    ----------
    Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
        Wiley. ISBN 978-1118840313.

    Fox J. and Weisberg, S. (2011) An R Companion to Applied Regression, Second Edition Sage.

    Mann–Whitney U test. (2017, June 20). In Wikipedia, The Free Encyclopedia.
        From https://en.wikipedia.org/w/index.php?title=Mann%E2%80%93Whitney_U_test&oldid=786593885

    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4


    """
    def __init__(self, y1, y2=None, group=None, continuity=True):

        if group is None:
            self.y1 = y1
            self.y2 = y2
        else:
            if len(np.unique(group)) > 2:
                raise ValueError('there cannot be more than two groups')

            obs_matrix = npi.group_by(group, y1)
            self.y1 = obs_matrix[1][0]
            self.y2 = obs_matrix[1][1]

        self.n1 = len(self.y1)
        self.n2 = len(self.y2)
        self.n = self.n1 + self.n2

        self.continuity = continuity
        self.ranks = self._rank()
        self.u_statistic = self._u()
        self.meanrank = self._mu()
        self.sigma = self._sigma_val()
        self.z_value = self._z()
        self.p_value = self._p_val()
        self.effect_size = self._eff_size()
        self.test_summary = {
            'continuity': self.continuity,
            'U': self.u_statistic,
            'mu meanrank': self.meanrank,
            'sigma': self.sigma,
            'z-value': self.z_value,
            'effect size': self.effect_size,
            'p-value': self.p_value,
            'test description': 'Mann-Whitney U test'
        }

    def _u(self):
        r"""
        Calculates the Mann-Whitney U statistic.

        Returns
        -------
        u : float

        Notes
        -----
        The chosen :code:`U` statistic is the smaller of the two statistics. The :code:`U`-statistic
        for sample :code:`k` is defined as:

        .. math::

            U_k = n_1 n_2 + \frac{n_k (n_k + 1)}{2} - \sum{R_k}

        Where :code:`n` is the number of sample observations and :code:`\sum{R_k}` is the sum of the
        ranked sample observations.

        The second sample :code:`U`-statistic can also be found by:

        .. math::

            U_1 + U_2 = n_1 n_2

        References
        ----------
        Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
            Wiley. ISBN 978-1118840313.

        """
        u1 = self.n1 * self.n2 + (self.n1 * (self.n1 + 1)) / 2. - np.sum(self.ranks)
        u2 = self.n1 * self.n2 - u1

        u = np.minimum(u1, u2)

        return u

    def _mu(self):
        r"""
        Computes the mean of the ranked sample observations.

        Returns
        -------
        mu_rank : float
            The mean of the ranked sample values.

        Notes
        -----
        The mean of the ranked samples is defined as:

        .. math::

            m_u = \frac{n_1 n_2}{2}

        References
        ----------
        Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
            Wiley. ISBN 978-1118840313.

        Mann–Whitney U test. (2017, June 20). In Wikipedia, The Free Encyclopedia.
            From https://en.wikipedia.org/w/index.php?title=Mann%E2%80%93Whitney_U_test&oldid=786593885

        """
        mu_rank = (self.n1 * self.n2) / 2. + (0.5 * self.continuity)

        return mu_rank

    def _sigma_val(self):
        r"""
        Calculates the standard deviation of the ranked sample observations.

        Returns
        -------
        sigma : float
            The standard deviation of the ranked sample values.

        Notes
        -----
        If there are no tied sample ranks, the standard deviation, :math:`\sigma_U`, can be calculated as
        the following:

        .. math::

            \sigma_U = \sqrt{\frac{n_1 n_2 (n_1 + n_2 + 1)}{12}}

        When tied ranks are present, the corrected standard deviation formula should be used instead.

        .. math::

            \sigma_{U_corr} = \sqrt{\frac{n_1 n_2}{12} \large((n + 1) - \sum_{i=1}^k \frac{t^3 - t_i}{n(n - 1)}\large)}

        Where :math:`n_1` and :math:`n_2` are the number of sample observations of group one and two, :math:`t_i` is
        the number of values sharing rank :math:`i` and :math:`k` is the number of unique ranks.

        References
        ----------
        Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
            Wiley. ISBN 978-1118840313.

        Mann–Whitney U test. (2017, June 20). In Wikipedia, The Free Encyclopedia.
            From https://en.wikipedia.org/w/index.php?title=Mann%E2%80%93Whitney_U_test&oldid=786593885

        """
        rankcounts = np.unique(self.ranks, return_counts=True)[1]

        sigma = np.sqrt(((self.n1 * self.n2) * (self.n + 1)) / 12. * (
                    1 - np.sum(rankcounts ** 3 - rankcounts) / float(self.n ** 3 - self.n)))

        return sigma

    def _z(self):
        r"""
        Computes the standardized :math:`z` value.

        Returns
        -------
        z : float
            The standardized value.

        Notes
        -----
        The standardized value is found by the following formula:

        .. math::

            z = \frac{U - m_u}{\sigma_u}

        References
        ----------
        Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
            Wiley. ISBN 978-1118840313.

        Mann–Whitney U test. (2017, June 20). In Wikipedia, The Free Encyclopedia.
            From https://en.wikipedia.org/w/index.php?title=Mann%E2%80%93Whitney_U_test&oldid=786593885

        """
        z = (np.absolute(self.u_statistic - self.meanrank)) / self.sigma

        return z

    def _p_val(self):
        r"""
        Returns the p-value.

        Returns
        -------
        p : float
            The computed p value.

        Notes
        -----
        When sample sizes are large enough (:math:`n > 20`), the distribution of :math:`U` is normally
        distributed.

        """
        p = 1 - norm.cdf(self.z_value)

        return p * 2

    def _eff_size(self):
        r"""
        Computes the effect size for determining the degree of association between groups.

        Returns
        -------
        es : float
            The effect size.

        Notes
        -----
        The effect size is defined as:

        .. math::

            ES = \frac{|z|}{\sqrt{n}}

        Effect sizes range from 0 to 1. A handy reference provided by Cohen (1988) defined the 'strength'
        of the effect size as:

        1. small = 0.10
        2. medium = 0.30
        3. large = 0.50

        References
        ----------
        Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
            Wiley. ISBN 978-1118840313.

        """
        es = np.abs(self.z_value) / np.sqrt(self.n)

        return es

    def _rank(self):
        ranks = np.concatenate((self.y1, self.y2))

        ranks = rankdata(ranks, 'average')

        ranks = ranks[:self.n1]

        return ranks


class MedianTest(object):
    r"""
    Performs Mood's Median test for k samples.

    Parameters
    ----------
    sample1, sample2, ... : array-like
        One-dimensional array-like objects (numpy array, list, pandas DataFrame or pandas Series) containing the
        observed sample data. Each sample may be of different lengths.
    ties : str, {'below', 'above', 'ignore'}
        Method for handling tied observations when sorting the observations into the above and below rows of the
        contingency table. If 'below' (default), values less than or equal to the median are added to the bottom
        row of the contingency table. If 'above', values less than to median are used.
    continuity : bool, default True
        If True, a continuity correction was applied when the Median test is performed. If False, no continuity
        correction is applied.
    posthoc : bool, default False
    names : array-like, default None

    Attributes
    ----------
    observation_vectors : list
        The passed observation vectors.
    combined_array : array-like
        One-dimensional array of all the observation vectors combined.
    grand_median : int
        Grand median of the arrays.
    n : int
        The total sample size.
    degrees_of_freedom : int
        Degrees of freedom, defined as the number of observations vectors - 1.
    ties : str
        The tie decision method.
    continuity : bool
        If True, a continuity correction was applied when the Median test was performed. If False, no continuity
        correction is applied.
    contingency_table : array-like
        The computed :math:`2 \times k` table of the number of samples above the grand median (in the first row) and
        below the grand median (second row).
    test_statistic : float
        The computed chi-square test statistic.
    p_value : float
        The associated p-value of the test statistic.
    posthoc : pandas DataFrame
    test_summary : dict
        A dictionary containing the test summary statistics including the contigency table, grand median, p-value, and
        test statistic.

    Raises
    ------
    ValueError
        Raised if the :code:`ties` parameter is not one of {'below' (default), 'above', 'ignore'}
    ValueError
        Raised the :code:`names` parameter does not have the same length as the number of observation vectors when
        performing a post-hoc test.

    Notes
    -----
    The median test, sometimes referred to as Mood's median test, is a nonparametric procedure for investigating
    whether the median of the populations from which :math:`k` sample groups are drawn is equal. The test is a special
    case of the chi-square test of dependence. The null and alternative hypotheses when employing the median test may
    be written similarly as:

    .. math::

        $H_0$: All $k$ populations have the same median.
        $H_A$: At least two of the $k$ populations have the different medians.

    Given :math:`k` samples with :math:`n_1, n_2, \cdots, n_k` data observations, the median test proceeds by computing
    the grand median of the combined observations. A :math:`2 \times k` contingency table is then constructed, where
    the top row contains the number of total observations above the grand median for each of the :math:`k` sample
    groups and the bottom row is the number of observations below the grand median. Ties between the individual
    observations and the grand median are either put in the top or bottom row, or discarded entirely. A chi-square test
    of independence is then performed on the constructed :math:`2 \times k` contingency table.

    The test statistic of the median test, typically denoted :math:`T`, is defined as:

    .. math

        T = \frac{N^2}{ab} \sum \frac{\left ( O_{1i} - \frac{n_i a}{N} \right )^2}{n_i}

    Where :math:`a` is the marginal total of the :math:`2 \times k` contingency table for observations above the grand
    median while $b$ is the marginal total for those observations below the grand median. The test statistic is assumed
    to have a chi-square distribution where the degrees of freedom is defined as :math:`k - 1`.

    Examples
    --------
    >>> g1 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
    >>> g2 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
    >>> g3 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]
    >>> m = MedianTest(g1, g2, g3)
    >>> m.test_summary
    {'contingency_table': array([[ 5, 10,  7],
                                [11,  5, 10]]),
     'grand median': 34.0,
     'p-value': 0.12609082774093244,
     'test_statistic': 4.141505553270259}

    References
    ----------
    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/meditest.htm

    https://psych.unl.edu/psycrs/handcomp/hcmedian.PDF

    Wikipedia contributors. (2017, June 27). Median test. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:23, August 19, 2018, from https://en.wikipedia.org/w/index.php?title=Median_test&oldid=787822318

    """
    def __init__(self, *args, ties='below', continuity=True, posthoc=False, names=None):
        self.observation_vectors = list([*args])

        combined_array = np.hstack(self.observation_vectors)
        self.grand_median = np.median(combined_array)
        self.n = combined_array.shape[0]

        self.degrees_of_freedom = len(self.observation_vectors) - 1

        if ties not in ('below', 'above', 'ignore'):
            raise ValueError("ties parameter must be one of 'below' (default), 'above', or 'ignore'")

        self.ties = ties
        self.continuity = continuity
        self.contingency_table = self._cont_table()
        self.test_statistic, self.p_value = self._chi_test()

        self.test_summary = {
            'test_statistic': self.test_statistic,
            'p-value': self.p_value,
            'grand median': self.grand_median,
            'contingency_table': self.contingency_table,
        }

        if posthoc:
            self.posthoc = self._multiple_comparisons(names=names)

        else:
            self.posthoc = 'None'

    def _cont_table(self):
        above = []
        below = []

        for vec in self.observation_vectors:
            vec_arr = np.array(vec)
            
            if self.ties == 'below':
                above.append(len(vec_arr[vec_arr > self.grand_median]))
                below.append(len(vec_arr[vec_arr <= self.grand_median]))
            
            elif self.ties == 'above':
                above.append(len(vec_arr[vec_arr >= self.grand_median]))
                below.append(len(vec_arr[vec_arr < self.grand_median]))
            
            else:
                vec_arr = vec_arr[vec_arr != self.grand_median]
                
                above.append(len(vec_arr[vec_arr > self.grand_median]))
                below.append(len(vec_arr[vec_arr < self.grand_median]))

        cont_table = np.vstack((above, below))

        return cont_table

    def _chi_test(self):
        c = ChiSquareContingency(self.contingency_table, continuity=self.continuity)

        return c.chi_square, c.p_value

    def _multiple_comparisons(self, names=None):
        if names is not None:
            if len(names) != len(self.observation_vectors):
                raise ValueError('group names array must be the same length as the number of sample groups.')

        else:
            names = []
            for i in range(0, len(self.observation_vectors)):
                names.append('Group {num}'.format(num=i))

        dat = dict(zip(names, self.observation_vectors))
        combs = [{j: dat[j] for j in i} for i in combinations(dat, 2)]

        group_comb = []
        t_stat = []
        p_val = []
        grand_med = []

        for comb in combs:
            name1, group1 = list(comb.keys())[0], list(comb.values())[0]
            name2, group2 = list(comb.keys())[1], list(comb.values())[1]

            m = MedianTest(group1, group2, names=[name1, name2])

            group_comb.append(str(name1) + ' : ' + str(name2))
            t_stat.append(m.test_statistic)
            p_val.append(m.p_value)
            grand_med.append(m.grand_median)

        result_df = pd.DataFrame({'groups': group_comb,
                                  'test statistic': t_stat,
                                  'p-value': p_val,
                                  'grand median': grand_med})

        return result_df


class RunsTest(object):
    r"""
    Performs the non-parametric one-sample runs test for determining if a sample is random.

    Parameters
    ----------
    x : array-like
        One-dimensional array-like (Pandas Series or DataFrame, Numpy array, or list)
        designating first sample observation values.
    continuity : bool, default True
        If True, continuity correction is applied when calculating the z-score.

    Attributes
    ----------
    x : array-like
        Numpy array of given data.
    runs : array-like
        Count and location of runs in given data.
    r : int
        The number of runs in specified data.
    test_summary : dict
        Dictionary containing relevant computed test statistics.

    Notes
    -----
    The runs test is a non-parametric test that examines the order or sequence of elements in a two-element
    (heads/tails, plus/minus, etc.) one-dimensional array to determine if the sample is random. For example, the
    following array of coin tosses has eight total 'runs'.

    .. math::

        H T H H T T H H H T H T

    When testing the randomness of small samples, the critical values of the test are determined from a critical
    value table. Small samples are typically defined as samples with each binary response not having equal to or more
    than 20 values. For example, the above array has :math:`n_1 = H = 7` and :math:`n_2 = T = 5` and thus would be
    designated as a small sample. Two critical value tables exist for the one-sample runs test. The first table,
    typically denoted :math:`F_1`, gives values of which are small enough that the probability associated with
    their occurrence under the null hypothesis :math:`H_0` is :math:`p = 0.025`. The second critical value table,
    typically denoted :math:`F_{11}` gives values of :math:`r` which are large enough that the probability associated
    with their occurrence under the null hypothesis is :math:`p = 0.025`. Thus, any observed value of the number of
    runs, :math:`r` is equal to or less than the value shown in :math:`F_1` or is equal to or larger than the value
    shown in :math:`F_{11}` is in the region of rejection. Critical values are given for :math:`\alpha = 0.05`.

    When the number of samples is large enough (each binary response having equal to or more than 20 responses), the
    sampling distribution becomes close enough to a normal distribution to use as an approximation.

    The mean of the sampling distribution :math:`\mu_r` is defined as:

    .. math::

        \mu_r = \frac{2n_1 n_2}{n_1 + n_2} + 1

    with variance of the sampling distribution :math:`\sigma^2` defined as:

    .. math::

        \sigma^2_r = \frac{2 n_1 n_2 (2n_1 n_2 - n_1 - n_2)}{(n_1 + n_2)^2 (n_1 + n_2 - 1)}

    Thus, a z-score can be computed to test the null hypothesis :math:`H_0`:

    .. math::

        z = \frac{r - \mu_r}{\sigma_r} = \frac{r - \large(\frac{2n_1 n_2}{n_1 + n_2} + 1 \large)}{\sqrt{\sigma^2_r = \frac{2 n_1 n_2 (2n_1 n_2 - n_1 - n_2)}{(n_1 + n_2)^2 (n_1 + n_2 - 1)}}}

    As the sample is approximately normally distributed, the critical value of the z-score can be found using the
    cumulative normal distribution function.

    If continuity correction is applied, the z-score is calculated as:

    .. math::

        z = \frac{|r - \mu_r| - 0.5}{\sigma_r}

    Where :math:`r`, :math:`\mu_r` and :math:`\sigma_r` are defined the same as above.

    Examples
    --------
    >>> s = ['m','f','m','f','m','m','m','f','f','m','f','m','f','m','f']
    >>> r = RunsTest(s)
    >>> r.r
    12
    >>> r.runs
    array([1, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1])
    >>> r.test_summary
    {'probability': 0.7672105672105671,
     'r critical value 1': 4,
     'r critical value 2': 13}

    References
    ----------
    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. "Wald–Wolfowitz runs test." Wikipedia, The Free Encyclopedia.
        Wikipedia, The Free Encyclopedia, 8 Jun. 2019. Web. 29 Sep. 2019.

    """
    def __init__(self, x, continuity=False):
        if not isinstance(x, np.ndarray):
            self.x = np.array(x)
        else:
            self.x = x

        self.runs, self.r = count_runs(self.x)
        self.continuity = continuity
        self.test_summary = self._runs_test()

    def _runs_test(self):
        r"""
        Primary method for performing the one-sample runs test.

        Returns
        -------
        dict
            Dictionary containing relevant test statistics of the one-sample runs test.

        """
        n1, n2 = Counter(self.x).values()

        r_range = np.arange(2, self.r + 1)
        evens = r_range[r_range % 2 == 0]
        odds = r_range[r_range % 2 != 0]

        p_even = 1 / comb(n1 + n2, n1) * np.sum(2 * comb(n1 - 1, evens / 2 - 1) * comb(n2 - 1, evens / 2 - 1))

        p_odd = 1 / comb(n1 + n2, n1) * np.sum(comb(n1 - 1, odds - 1) * comb(n2 - 1, odds - 2) +
                                               comb(n1 - 1, odds - 2) * comb(n2 - 1, odds - 1))

        p = p_even + p_odd

        if all(np.array([n1, n2]) <= 20):
            r_crit_1, r_crit_2 = r_critical_value(n1, n2)

            test_summary = {
                'probability': p,
                'r critical value 1': r_crit_1,
                'r critical value 2': r_crit_2
            }
            return test_summary

        else:
            mean = (2 * n1 * n2) / (n1 + n2) + 1
            sd = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
            z = (np.abs(self.r - mean) - self.continuity * 0.5) / sd
            p_val = norm.sf(z) * 2

            test_summary = {
                'probability': p,
                'mean of runs': mean,
                'standard deviation of runs': sd,
                'z-value': z,
                'p-value': p_val,
                'continuity': self.continuity
            }

            return test_summary


class SignTest(object):
    r"""
    Computes the nonparametric sign test of differences between paired observations.

    Parameters
    ----------
    x : array-like
    y : array-like, optional
    alternative : str, {'two-sided', 'greater', 'less'}

    Attributes
    ----------
    x : array-like
    y : array-like
    alternative : str, {'two-sided', 'greater', 'less'}
    n : int
    sample_differences : array-like
    sample_differences_median : float
    difference_counts : dict
    p_value : float
    test_summary : dict

    Notes
    -----

    Examples
    --------
    >>> f = [4, 4, 5, 5, 3, 2, 5, 3, 1, 5, 5, 5, 4, 5, 5, 5, 5]
    >>> m = [2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 5, 2, 5, 3, 1]
    >>> s = SignTest(f, m)
    >>> s.test_summary
    {'differences count': {'negative': 3, 'positive': 11, 'ties': 3},
     'median difference': 2.0,
     'p-value': 0.0286865234375}

    References
    ----------
    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2018, July 25). Sign test. In Wikipedia, The Free Encyclopedia.
        Retrieved 14:52, August 23, 2018, from https://en.wikipedia.org/w/index.php?title=Sign_test&oldid=851943717

    """
    def __init__(self, x, y=None, alternative='two-sided'):
        if not isinstance(x, np.ndarray):
            self.x = np.array(x)
        else:
            self.x = x

        if self.x.ndim >= 2:
            raise ValueError('x must not have more than two columns.')

        if self.x.ndim == 1 and y is None:
            raise ValueError('sample y must be passed if x does not contain two columns.')

        if self.x.ndim == 2:
            self.x = self.x[:, 0]
            self.y = self.x[:, 1]
        else:
            if not isinstance(y, np.ndarray):
                self.y = np.array(y)
            else:
                self.y = y

            if self.x.shape[0] != self.y.shape[0]:
                raise ValueError('x and y must have the same length.')

        if alternative not in ('two-sided', 'greater', 'less'):
            raise ValueError("'alternative must be one of 'two-sided' (default), 'greater', or 'less'.")

        self.alternative = alternative
        self.n = self.x.shape[0]
        self.sample_differences = self.x - self.y
        self.sample_differences_median = np.median(self.sample_differences)
        self.sample_sign_differences = np.sign(self.sample_differences)
        self.differences_counts = {
            'positive': np.sum(self.sample_sign_differences == 1),
            'negative': np.sum(self.sample_sign_differences == -1),
            'ties': np.sum(self.sample_sign_differences == 0)
        }

        self.p_value = self._sign_test()

        self.test_summary = {
            'p-value': self.p_value,
            'median difference': self.sample_differences_median,
            'differences count': self.differences_counts
        }

    def _sign_test(self):
        pos, neg = self.differences_counts['positive'], self.differences_counts['negative']

        n = pos + neg

        res = BinomialTest(n=int(n), x=int(pos), alternative=self.alternative)

        return res.p_value


class VanDerWaerden(object):
    r"""
    Performs the Van Der Waerden (normal scores) test for testing if k groups have the same distribution
    function.

    Parameters
    ----------
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector.
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector.
    alpha : float, default 0.05
        Desired alpha level for testing for significance.
    post-hoc : bool, default True
        If True, a post-hoc multiple comparisons test is performed.

    Attributes
    ----------
    design_matrix : array-like
        Numpy ndarray representing the data matrix for the analysis.
    ranked_matrix : array-like
        Numpy ndarray representing the data matrix with ranked observations.
    normal_score_matrix : array-like
        Numpy ndarray representing the data matrix with ranked observations and computed normal test scores.
    average_scores : list
        List of tuples containing each group name and its respective average normal score.
    test_statistic : float
        The computed Van Der Waerden test statistic, denoted :math:`T_1`.
    p_value : float
        The p-value of the calculated :math:`T_1` test statistic.
    alpha : float
        Desired alpha level for testing for significance.
    group : array-like
        One-dimensional numpy array of the passed or coerced group array.
    n : int
        Number of total observations.
    k : int
        Number of groups
    test_description: str
        Test performed.
    test_summary : dict
        Dictionary of test results.

    Notes
    -----
    The Van der Waerden test is a non-parametric test for testing the assumption that :math:`k` sample distribution
    functions are equal. Van der Waerden's test is similar to the Kruskal-Wallis one-way analysis of variance test in
    that it converts the data to ranks and then to standard normal distribution quantiles which are designated as the
    'normal scores'.

    The benefit of Van der Waerden's test is that it is performant compared to ANOVA (analysis of variance) when the
    samples are normally distributed and the Kruskal-Wallis test when the samples are not normally distributed.

    The null and alternative hypotheses of the Van der Waerden test can be stated generally as follows:

    :math:`H_0`: All of the :math:`k` population distribution functions are equal
    :math:`H_A`: At least one of the :math:`k` population distribution functions are not equal and tend to yield larger
    observations to the other distribution functions.

    Let :math:`n_j`, be the number of samples for each of the :math:`k` groups where :math:`j` is the j-th group.
    :math:`N` is the number of total samples in all groups, while :math:`X_{ij}` is the i-th value of the j-th group.
    The normal scores used in the Van der Waerden test are calculated as:

    .. math::

        A_{ij} = \Phi^{-1} \left( \frac{R \left( X_{ij} \right)}{N + 1} \right)

    where :math:`R(X_{ij})` and :math:`phi^{-1}` are the ranks of the :math:`X_{ij}` observation and the normal
    quantile function (percent point function), respectively. The average normal scores can then be calculated as:

    .. math::

        \bar{A}_j = \frac{1}{n_j} \sum^{n_j}_{i=1} A_{ij} \qquad j = 1, 2, \cdots, k

    The variance :math:`s^2` of the normal scores is defined as:

    .. math::

        s^2 = \frac{1}{N - 1} \sum^k_{i=1} \sum^{n_i}_{j=1} A^2_{ij}

    The Van der Waerden test statistic, :math:`T_1` is defined as:

    .. math::

        T_1 = \frac{1}{s^2} \sum^k_{i=1} n_i (\bar{A}_i)^2

    As the test is approximate to a chi-square distribution, the critical region for a significance level :math:`\alpha`
    is:

    .. math::

        T_1 = \chi^2_{\alpha, k-1}

    When the null hypothesis is rejected (p-value within the critical region) and at least one of the sample
    distribution functions differs, a post-hoc multiple comparions test can be performed to get a better sense of
    which populations differ from the others. Two sample populations, :math:`j_1` and :math:`j_2`, tend to be different
    if the following is true:

    .. math::

        | \bar{A}_{j_1} - \bar{A}_{j_2} | > st_{1-\alpha/2} \sqrt{\frac{N-1-T_1}{N-k}} \sqrt{\frac{1}{n_{j_1}} + \frac{1}{n_{j_2}}}

    Examples
    --------

    References
    ----------
    Conover, W. J. (1999). Practical Nonparameteric Statistics (Third ed.). Wiley.

    Wikipedia contributors. "Van der Waerden test." Wikipedia, The Free Encyclopedia.
        Wikipedia, The Free Encyclopedia, 8 Feb. 2017. Web. 8 Mar. 2020.

    """
    def __init__(self, *args, group=None, alpha=0.05, post_hoc=True):
        if group is not None and len(args) > 1:
            raise ValueError('Only one sample vector should be passed when including a group vector')

        self.design_matrix = _build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.alpha = alpha
        self.n = self.design_matrix.shape[0]
        self.k = len(np.unique(self.design_matrix[:, 0]))
        self._group_obs = np.array([i[1] for i in
                                    npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], len)])

        self.ranked_matrix = _rank(self.design_matrix)
        self.normal_score_matrix = self._normal_scores()
        self.average_scores = self._normal_scores_average()
        self.score_variance = self._normal_scores_variance()
        self.test_statistic, self.p_value = self._test_statistic()
        #self.minimum_significant_difference = self._min_significant_difference()
        self.test_description = 'Van Der Waerden (normal scores) test'
        self.test_summary = {'test_description': self.test_description,
                             'test_statistic': self.test_statistic,
                             #'least_significant_difference': self.minimum_significant_difference,
                             'p_value': self.p_value
                             }

        if post_hoc:
            self.multiple_comparisons = self._post_hoc()
            self.test_summary['post_hoc'] = self.multiple_comparisons
        else:
            self.multiple_comparisons = None

    def _normal_scores(self):
        r"""
        Calculates the normal scores used in the Van der Waerden test.

        Returns
        -------
        score_matrix : array-like
        Numpy ndarray representing the data matrix with ranked observations and computed normal test scores.

        Notes
        -----
        Let :math:`n_j`, be the number of samples for each of the :math:`k` groups where :math:`j` is the j-th group.
        :math:`N` is the number of total samples in all groups, while :math:`X_{ij}` is the i-th value of the j-th
        group. The normal scores used in the Van der Waerden test are calculated as:

        .. math::

            A_{ij} = \phi^{-1} \left( \frac{R \left( X_{ij} \right)}{N + 1} \right)

        References
        ----------
        Conover, W. J. (1999). Practical Nonparameteric Statistics (Third ed.). Wiley.

        Wikipedia contributors. "Van der Waerden test." Wikipedia, The Free Encyclopedia.
            Wikipedia, The Free Encyclopedia, 8 Feb. 2017. Web. 8 Mar. 2020.

        """
        aij = norm.ppf(list(self.ranked_matrix[:, 2] / (self.n + 1)))
        score_matrix = np.column_stack([self.ranked_matrix, aij])

        return score_matrix

    def _normal_scores_average(self):
        r"""
        Returns the average normal scores for each group.

        Returns
        -------
        average_scores : list
            List of tuples containing each group name and its respective average normal score.

        Notes
        -----
        The average normal scores for each group are computed as, where :math:`A_{ij}` are the computed normal scores.

        .. math::

            \bar{A}_j = \frac{1}{n_j} \sum^{n_j}_{i=1} A_{ij} \qquad j = 1, 2, \cdots, k

        References
        ----------
        Conover, W. J. (1999). Practical Nonparameteric Statistics (Third ed.). Wiley.

        Wikipedia contributors. "Van der Waerden test." Wikipedia, The Free Encyclopedia.
            Wikipedia, The Free Encyclopedia, 8 Feb. 2017. Web. 8 Mar. 2020.

        """
        average_scores = npi.group_by(self.normal_score_matrix[:, 0], self.normal_score_matrix[:, 3], np.mean)

        return average_scores

    def _normal_scores_variance(self):
        r"""
        Calculates the variance of the normal scores.

        Returns
        -------
        score_variance : float
            The variance of the computed normal scores.

        Notes
        -----
        The variance :math:`s^2` of the normal scores is defined as:

        .. math::

            s^2 = \frac{1}{N - 1} \sum^k_{i=1} \sum^{n_i}_{j=1} A^2_{ij}

        References
        ----------
        Conover, W. J. (1999). Practical Nonparameteric Statistics (Third ed.). Wiley.

        Wikipedia contributors. "Van der Waerden test." Wikipedia, The Free Encyclopedia.
            Wikipedia, The Free Encyclopedia, 8 Feb. 2017. Web. 8 Mar. 2020.

        """
        score_variance = np.sum(self.normal_score_matrix[:, 3] ** 2) / (self.n - 1)

        return score_variance

    def _test_statistic(self):
        r"""
        Returns the Van der Waerden test statistic, :math:`T_1` and the associated p-value.

        Returns
        -------
        t1 : float
            The Van der Waerden test statistic
        p_value : float
            The computed p-value

        Notes
        -----
        The Van der Waerden test statistic, :math:`T_1` is defined as:

        .. math::

            T_1 = \frac{1}{s^2} \sum^k_{i=1} n_i (\bar{A}_i)^2

        References
        ----------
        Conover, W. J. (1999). Practical Nonparameteric Statistics (Third ed.). Wiley.

        Wikipedia contributors. "Van der Waerden test." Wikipedia, The Free Encyclopedia.
            Wikipedia, The Free Encyclopedia, 8 Feb. 2017. Web. 8 Mar. 2020.

        """
        average_scores = np.array([i for _, i in self.average_scores])
        t1 = np.sum(self._group_obs * average_scores ** 2) / self.score_variance

        p_value = chi2.sf(t1, self.k - 1)

        return t1, p_value

    # def _min_significant_difference(self):
    #     mse = self.score_variance * ((self.n - 1 - self.test_statistic) / (self.n - self.k))
    #
    #     msd = t.ppf(1 - self.alpha / 2, self.n - self.k) * np.sqrt(2 * mse / self.k)
    #
    #     return msd

    def _post_hoc(self):
        r"""
        Returns a pandas DataFrame containing the multiple comparison test results.

        Returns
        -------
        groups : pandas DataFrame
            pandas DataFrame containing results of the multiple comparisons test.

        Notes
        -----
        When the null hypothesis is rejected (p-value within the critical region) and at least one of the sample
        distribution functions differs, a post-hoc multiple comparions test can be performed to get a better sense of
        which populations differ from the others. Two sample populations, :math:`j_1` and :math:`j_2`, tend to be different
        if the following is true:

        .. math::

            | \bar{A}_{j_1} - \bar{A}_{j_2} | > st_{1-\alpha/2} \sqrt{\frac{N-1-T_1}{N-k}} \sqrt{\frac{1}{n_{j_1}} + \frac{1}{n_{j_2}}}

        References
        ----------
        Conover, W. J. (1999). Practical Nonparameteric Statistics (Third ed.). Wiley.

        Wikipedia contributors. "Van der Waerden test." Wikipedia, The Free Encyclopedia.
            Wikipedia, The Free Encyclopedia, 8 Feb. 2017. Web. 8 Mar. 2020.

        """
        average_scores = [i for _, i in self.average_scores]

        sample_sizes = 1 / np.array(list(combinations(self._group_obs, 2)))[:, 0] + \
                       1 / np.array(list(combinations(self._group_obs, 2)))[:, 1]

        average_score_differences = np.abs(np.array(list(combinations(average_scores, 2)))[:, 0] - \
                                    np.array(list(combinations(average_scores, 2)))[:, 1])

        group_names = np.unique(self.design_matrix[:, 0])

        groups = pd.DataFrame(np.array(list(combinations(group_names, 2))))

        groups['groups'] = groups[0] + ' - ' + groups[1]
        groups['score'] = average_scores

        groups['difference'] = average_score_differences > np.sqrt(self.score_variance) * \
                               t.ppf(1 - self.alpha / 2, self.n - self.k) * \
                               np.sqrt((self.n - 1 - self.test_statistic) / (self.n - self.k)) * np.sqrt(sample_sizes)

        del groups[0]
        del groups[1]

        return groups


class WaldWolfowitz(object):
    r"""
    Performs the Wald-Wolfowitz Two-Sample runs test for two independent samples.

    Parameters
    ----------
    x : array-like
        First sample observation vector.
    y : array-like
        Second sample observation vector.
    continuity: bool, default True
        If True, continuity correction is applied during the Wald-Wolfowitz test procedure.

    Attributes
    ----------
    x : array-like
        First sample observation vector.
    y : array-like
        Second sample observation vector.
    continuity: bool
        If True, continuity correction is applied during the Wald-Wolfowitz test procedure.
    runs : int
        The number of total runs in the ranked and ordered samples.
    probability : float
        The estimated proability of getting an observed value of :math:`r` or smaller.
    z : float
        The computed z-score.
    test_summary: dict
        Dictionary containing relevant test summary statistics.

    Examples
    --------
    >>> x = [20, 55, 29, 24, 75, 56, 31, 45]
    >>> y = [23, 8, 24, 15, 8, 6, 15, 15, 21, 23, 16, 15, 24, 15, 21, 15, 18, 14, 22, 15, 14]
    >>> w = nonparametric.WaldWolfowitz(x=x, y=y)
    >>> w.test_summary
    {'probability': 0.051136902411265235,
     'runs': 6,
     'mean of runs': 12.586206896551724,
     'standard deviation of runs': 2.0929642628266922,
     'z-value': 2.907936367882308,
     'p-value': 0.0018191117963075613,
     'continuity': True}

    Notes
    -----
    The Wald-Wolfowitz runs test is used to test the hypothesis that two independent samples have been drawn from the
    same population rejects the hypothesis if the two samples differ in any way. The Wald-Wolfowitz test is useful in
    that it can be used to test if two samples differ in more than one respect, whether that be central tendency,
    variance, skewness, kurtosis, and so on.

    Before the test is performed, the two samples are ordered and ranked, keeping their group membership but as a
    single array. Depending on the sample sizes of both samples, the test either uses a critical value table or
    is approximated using a normal distribution.

    The sampling distribution of the observed runs :math:`r` stems from when the two samples are ordered into a single
    array, the total number of possible arrangements becomes binomial.

    .. math::

        \binom{n_1 + n_2}{n_1} = \binom{n_1 + n_2}{n_2}

    It can then be shown that the probability of getting an observed value of the runs :math:`r` or a smaller value
    when the value of :math:`r` is even is:

    .. math::

        p(r \geq r^{\prime}) = \frac{1}{\binom{n_1 + n_2}{n_1}} \sum^{r^{\prime}}_{r=2} (2) \binom{n_1 - 1}{\frac{r}{2} - 1} \binom{n_2 - 1}{\frac{r}{2} - 1}

    When :math:`r` is odd, the probability is defined as:

    .. math::

        p(r \geq r^{\prime}) = \frac{1}{\binom{n_1 + n_2}{n_1}} \sum^{r^{\prime}}_{r=2} \Bigg[ \binom{n_1 - 1}{k - 1} \binom{n_2 - 1}{k - 2} + \binom{n_1 - 1}{k - 2} \binom{n_2 - 1}{k - 1} \Bigg]

    where :math:`r = 2k - 1`

    In the case of small samples, (:math:`n_1, n_2 \geq 20`), a critical value table is used to determine the
    significance at a alpha of 0.05. For example, if the observed runs value, :math:`r` is equal to or less than the
    corresponding value in the critical value table, the null hypothesis :math:`H_0` may be rejected at a significance
    level of 0.05. Conversely, if the observed :math:`r` value is greater than the corresponding value in the table,
    then the null hypothesis cannot be rejected.

    When the small sample case does not apply, the sampling distribution of :math:`r` under the null hypothesis is
    approximately normal.

    The mean is defined as:

    .. math::

        \mu_r = \frac{2n_1 n_2}{n_1 + n_2} + 1

    With standard deviation:

    .. math::

        \sigma_r = \sqrt{\frac{2n_1n_2(2n_1n_2 - n_1 - n_2)}{(n_1 + n_2)^2(n_1 + n_2 - 1)}}

    The z-score, :math:`z = \frac{r - \mu_r}{\sigma_r}, can then be defined as:

    .. math::

        z = \frac{r - \large(\frac{2n_1 n_2}{n_1 + n_2} + 1 \large)}{\sqrt{\frac{2n_1n_2(2n_1n_2 - n_1 - n_2)}{(n_1 + n_2)^2(n_1 + n_2 - 1)}}}

    In the case of large samples, the sampling distribution is normally distributed with zero mean and variance.

    When the large sample setting applies but the total sample size :math:`N = (n_1 + n_2)` is still not quite large
    (large unfortunately still being somewhat subjective; however, generally this implies that the sample size is not
    large enough for the assumption of an approximately normally distributed sample to hold), a continuity correction
    is recommended (and in some cases required). The continuity correction is performed by subtracting :math:`0.5`
    from the absolute difference between the observed runs :math:`r` and the mean :math:`\mu_r` in the z-score computation.

    .. math::

        z = \frac{|r - \mu_r| - .5}{\sigma_r}

    References
    ----------
    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    Wikipedia contributors. (2018, August 22). Wald–Wolfowitz runs test. In Wikipedia, The Free Encyclopedia.
        Retrieved 13:54, September 13, 2018,
        from https://en.wikipedia.org/w/index.php?title=Wald%E2%80%93Wolfowitz_runs_test&oldid=856082551

    """
    def __init__(self, x, y, continuity=True):
        self.x, self.y = x, y
        self.n1, self.n2 = len(x), len(y)
        self.a = np.sort(np.array(self.x + self.y))
        self.continuity = continuity
        self.r, self.test_summary = self._test()
        self.p_value = self.test_summary['p-value']
        self.probability = self.test_summary['probability']
        self.description = 'Wald-Wolfowitz Runs Test for Two Independent Samples'

        try:
            self.z = self.test_summary['z-value']
        except KeyError:
            pass

    def _test(self):
        a = pd.DataFrame({'a': list(np.repeat('A', len(self.x))), 'b': self.x})
        b = pd.DataFrame({'a': list(np.repeat('B', len(self.y))), 'b': self.y})
        c = a.append(b).sort_values('b')['a']

        r = count_runs(c)[1]
        r_range = np.arange(2, r + 1)
        evens = r_range[r_range % 2 == 0]
        odds = r_range[r_range % 2 != 0]

        p_even = 1 / comb(self.n1 + self.n2, self.n1) * \
                 np.sum(2 * comb(self.n1 - 1, evens / 2 - 1) *
                        comb(self.n2 - 1, evens / 2 - 1))

        p_odd = 1 / comb(self.n1 + self.n2, self.n1) * \
                np.sum(comb(self.n1 - 1, odds - 1) *
                       comb(self.n2 - 1, odds - 2) +
                       comb(self.n1 - 1, odds - 2) * comb(self.n2 - 1, odds - 1))

        p = p_even + p_odd

        if all(np.array([self.n1, self.n2]) <= 20):
            r_crit_1, r_crit_2 = r_critical_value(self.n1, self.n2)

            test_summary = {
                'probability': p,
                'runs': r,
                'r critical value 1': r_crit_1,
                'r critical value 2': r_crit_2
            }
        else:
            mean = (2 * self.n1 * self.n2) / (self.n1 + self.n2) + 1
            sd = np.sqrt((2 * self.n1 * self.n2 * (2 * self.n1 * self.n2 - self.n1 - self.n2)) /
                         ((self.n1 + self.n2) ** 2 * (self.n1 + self.n2 - 1)))
            z = (np.abs(r - mean) - self.continuity * 0.5) / sd
            p_val = norm.sf(z)

            test_summary = {
                'probability': p,
                'runs': r,
                'mean of runs': mean,
                'standard deviation of runs': sd,
                'z-value': z,
                'p-value': p_val,
                'continuity': self.continuity
            }

        return r, test_summary


class WilcoxonTest(object):
    r"""
    Performs Wilcoxon Rank Sum tests for matched pairs and independent samples.

    Parameters
    ----------
    y1 : array-like
        One-dimensional array-like (Pandas Series or DataFrame, Numpy array, or list)
        designating first sample observation values.
    y2 : array-like, optional
        One-dimensional array-like (Pandas Series or DataFrame, Numpy array, or list)
        designating second sample observation values.
    paired : bool, optional
        If True, performs a paired Wilcoxon Rank Sum test.
    mu : float, optional
        Optional parameter to specify the value to form the null hypothesis.

    Attributes
    ----------
    y1 : array-like
        First sample observation vector.
    y2 : array-like or None
        Second sample observation vector, if passed. Otherwise, will return None.
    n : int
        Number of sample observations.
    V : float
        Wilcoxon :math:`V`-statistic (also denoted :math:`W` and :math:`U` in some literature).
    z : float
        The standardized z-statistic.
    p : float
        p-value.
    effect_size : float
        The estimated effect size.

    Notes
    -----
    The Wilcoxon Rank Sum test is the nonparametric equivalent to a matched pairs or independent sample
    t-test and is also closely related to the Mann Whitney U-test for independent samples. In fact, the
    Wilcoxon Rank Sum test for two independent samples is equivalent to the Mann Whitney U-test. The
    respective test statistics :math:`W` (Mann-Whitney) and :math:`U` (Wilcoxon Rank Sum) are related
    in the following way:

    .. math::

        U = W - \frac{n_1 (n_1 + 1)}{2}

    The test procedure can be summarized into the following steps:

    1. If the test is for an independent sample, the observations are subtracted by the true mean of
    the null hypothesis :math:`mu` to obtain the signed differences. In the case of a paired test, the
    signed difference between each matched observation vector is found.
    2. The signed differences, typically denoted :math:`d_i`, are then ranked. Ties receive the average of
    the tied ranks.
    3. The test statistic :math:`V` (or :math:`T` in some literature) is then computed by assigning a
    :math:`1` for ranked values where the corresponding matched pair difference is positive or a
    :math:`0` for ranked values with a negative corresponding matched pair difference. These values are then
    summed to obtain the test statistic.
    4. The calculated test statistic can then be used to determine the significance of the observed value.

    When two sample observation vectors are passed into the :code:`wilcoxon_test` function with the parameter
    :code:`paired = False`, the Mann-Whitney U-test is performed.

    Examples
    --------
    The data used in this example is a subset of the professor salary dataset found in Fox and
    Weisberg (2011).

    >>> professor_salary = [139750, 173200, 79750, 11500, 141500,
    ...                     103450, 124750, 137000, 89565, 102580]
    >>> w = WilcoxonTest(professor_salary)
    >>> w.test_summary
    {'V': 55.0,
     'effect size': 0.8864052604279182,
     'p-value': 0.005062032126267768,
     'test description': 'Wilcoxon signed rank test',
     'z-value': 2.8030595529069404}

    References
    ----------
    Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
        Wiley. ISBN 978-1118840313.

    Fox J. and Weisberg, S. (2011) An R Companion to Applied Regression, Second Edition Sage.

    Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
        McGraw-Hill. ISBN 07-057348-4

    """
    def __init__(self, y1, y2=None, paired=True, mu=0, alpha=0.05, alternative='two-sided'):
        self.paired = paired
        self.median = mu
        self.alternative = alternative
        self.test_description = 'Wilcoxon signed rank test'

        if paired:
            if y2 is None:
                self.y1 = y1

            else:
                if len(y1) != len(y2):
                    raise ValueError('samples must have same length for paired test')

                self.y1 = np.array(y1) - np.array(y2)

        else:
            self.y1 = np.array(y1)

        self.n = len(self.y1)

        self.V = self._v_statistic()

        self.z = self._zvalue()
        self.p = self._pvalue()
        self.effect_size = self._eff_size()
        self.test_summary = {
            'V': self.V,
            'z-value': self.z,
            'p-value': self.p,
            'effect size': self.effect_size,
            'test description': self.test_description
        }

        # if self.n > 10:
        #     self.z = self._zvalue()
        # else:
        #     self.alpha = alpha
        #     self.alternative = alternative
        #
        #     if self.alternative == 'two-sided':
        #         alt = 'two-tail'
        #     else:
        #         alt = 'one-tail'
        #
        #     w_crit = w_critical_value(self.n, self.alpha, alt)

    def _v_statistic(self):
        r"""
        Computes the Wilcoxon test :math:`V`-statistic.

        Returns
        -------
        v : float
            The computed Wilcoxon test statistic.

        Notes
        -----
        The procedure to calculate :math:`V` can be summarized as the following:

        1. If the test is for an independent sample, the observations are subtracted by the true mean of
            the null hypothesis :math:`mu` to obtain the signed differences. In the case of a paired test, the
            signed difference between each matched observation vector is found.
        2. The signed differences, typically denoted :math:`d_i`, are then ranked. Ties receive the average of
            the tied ranks.
        3. The test statistic :math:`V` is then computed by assigning a :math:`1` for ranked values where the
            corresponding matched pair difference is positive or a :math:`0` for ranked values with a negative
            corresponding matched pair difference. These values are then
            summed to obtain the test statistic.

        More formally, the computation of the test statistic for a matched pair test can be written as:

        .. math::

            V = \sum_{i=1}^{N_r} \left[ sgn(x_{2,i} - x_{1,i}) R_i \right]

        For an independent sample test, the computation is written as:

        .. math::

            V = \sum_{i=1}^{N_r} \left[ sgn(x_i - \mu) R_i \right]

        Where :math:`\mu` is the value of the null hypothesis, :math:`H_0`.

        The test statistic :math:`V` is also referred to as :math:`T` in some literature.

        References
        ----------
        Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
            Wiley. ISBN 978-1118840313.

        Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
            McGraw-Hill. ISBN 07-057348-4

        """
        if self.paired:
            y_median_signed = self.y1
        else:
            y_median_signed = self.y1 - self.median

        y_median_unsigned = np.absolute(y_median_signed)

        ranks_signed = rankdata(y_median_signed, 'average')
        ranks_unsigned = rankdata(y_median_unsigned, 'average')

        z = np.where(ranks_signed > 0, 1, 0)

        v = np.sum(np.multiply(ranks_unsigned, z))

        return v

    def _zvalue(self):
        r"""
        Calculates the :math:`z`-score.

        Returns
        -------
        z : float
            The computed :math:`z`-score of the :math:`V`-statistic.

        Notes
        -----
        For larger sample sizes, :math:`N_r \geq 25`, (some literature states sample sizes of :math:`N_r \geq 10`
        is enough), the distribution of the :math:`V`-statistic converges to a normal distribution and thus a
        :math:`z`-score can be calculated.

        The :math:`z`-score is calculated as:

        .. math::

            z = \frac{V}{\sigma_V}

        Where :math:`\sigma_V` is the standard deviation of the distribution, which can be computed as:

        .. math::

            \sigma_V = \sqrt{\frac{N_r (N_r + 1)(2 N_r + 1)}{6}}

        References
        ----------
        Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
            Wiley. ISBN 978-1118840313.

        Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
            McGraw-Hill. ISBN 07-057348-4

        """
        sigma_w = np.sqrt((self.n * (self.n + 1) * (2 * self.n + 1)) / 6.)

        z = self.V / sigma_w

        return z

    def _pvalue(self):
        r"""
        Calculates the p-value.

        Returns
        -------
        p : float
            The calculated p-value

        Notes
        -----

        References
        ----------
        Siegel, S. (1956). Nonparametric statistics: For the behavioral sciences.
            McGraw-Hill. ISBN 07-057348-4

        """
        p = (1 - norm.cdf(np.abs(self.z)))

        if self.alternative == 'two-sided':
            p *= 2
        elif self.alternative == 'greater':
            p = 1 - p

        if p == 0:
            p = np.finfo(float).eps

        return p

    def _eff_size(self):
        r"""
        Computes the effect size for determining degree of association.

        Returns
        -------
        es : float
            The calculated effect size.

        Notes
        -----
        The effect size is defined as:

        .. math::

            ES = \frac{|z|}{\sqrt{N_r}}

        References
        ----------
        Corder, G.W.; Foreman, D.I. (2014). Nonparametric Statistics: A Step-by-Step Approach.
            Wiley. ISBN 978-1118840313.

        """
        es = np.abs(self.z) / np.sqrt(self.n)

        return es


def tie_correction(rank_array):
    r"""
    Computes the tie correction factor used in nonparametric statistical tests.

    Parameters
    ----------
    rank_array : array-like
        1-d array (numpy array, list, pandas DataFrame or Series) of ranks.

    Returns
    -------
    corr : float
        The correction factor for :math:`H` (or :math:`U` for the Mann-Whitney U-test).

    Notes
    -----
    The tie correction factor is defined as:

    .. math::

            1 - \frac{\sum_{t=1}^G (t_i^3 - t_i)}{N^3 - N}

    Where :code:`G` is the number of groups of tied ranks and :code:`t_i` is the number of
    tied values within the :code:`i^{th}` group.

    Examples
    --------
    The ranked values of an observation vector can be easily found using Scipy's :code:`tiecorrect`
    function.

    >>> obs = [4.17, 5.58, 5.18, 4.81, 4.17, 4.41, 5.31, 5.12, 5.54]
    >>> ranked_obs = rankdata(obs)
    >>> ranked_obs
    array([1.5, 9. , 6. , 4. , 1.5, 3. , 7. , 5. , 8. ])

    >>> tie_correction(ranked_obs)
    0.9916666666666667

    References
    ----------
    Wikipedia contributors. (2018, May 21). Kruskal–Wallis one-way analysis of variance.
            In Wikipedia, The Free Encyclopedia. From
            https://en.wikipedia.org/w/index.php?title=Kruskal%E2%80%93Wallis_one-way_analysis_of_variance&oldid=842351945

    """
    tied_groups = np.unique(rank_array, return_counts=True)[1]
    tied_groups = tied_groups[tied_groups > 1]

    corr = 1 - np.sum((tied_groups ** 3 - tied_groups)) / (rank_array.shape[0] ** 3 -
                                                           rank_array.shape[0])

    return corr


def count_runs(x, index=1):
    r"""

    Parameters
    ----------
    x : array-like
    index : int, default 1

    Returns
    -------

    Examples
    --------

    """
    runs = np.array([sum(1 for _ in r) for _, r in groupby(np.array(x))])

    run_count = len(runs)

    return runs, run_count
