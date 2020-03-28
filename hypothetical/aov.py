# encoding=utf8

"""
Functions for performing one-way analysis of variance (ANOVA) and multivariate analysis of variance (MANOVA).

One-Way Analysis of Variance
----------------------------

.. autosummary::
    :toctree: generated/

    AnovaOneWay
    BartlettsTest
    LevenesTest
    ManovaOneWay

References
----------
Andrews, D. F., and Herzberg, A. M. (1985), Data, New York: Springer-Verlag.

Dobson, A. J. (1983) An Introduction to Statistical Modelling.
    London: Chapman and Hall.

Fox J. and Weisberg, S. (2011) An R Companion to Applied Regression, Second Edition Sage.

Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
    Brigham Young University: John Wiley & Sons, Inc.

"""

import numpy as np
import numpy_indexed as npi
from scipy.stats import f, chi2

from hypothetical._lib import _build_des_mat
from hypothetical.descriptive import var


class AnovaOneWay(object):
    r"""
    Performs one-way ANOVA. One-way ANOVA (Analysis of Variance) is used to analyze and test
    the differences of two or more groups have the same population mean.

    Parameters
    ----------
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector.
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector(s).

    Attributes
    ----------
    design_matrix : array-like
        Numpy ndarray representing the data matrix for the analysis.
    group_names: array-like
        Numpy array of the group names.
    k : int
        The number of groups.
    group_degrees_of_freedom : int
        The group degrees of freedom, :code:`k - 1`.
    residual_degrees_of_freedom : int
        The residual degrees of freedom, "code:`n - k`.
    group_sum_squares : float
        The computed group (treatment) sum of squares, typically denoted :math:`SST` or :math:`SSH`.
    group_mean_squares : float
        The 'within' sample, or treatment, mean sum of squares, typically denoted :math:`MST` or :math:`MSH`.
    residual_sum_squares : float
        The residual sum of squares, usually denoted :math:`SSE`.
    residual_mean_squares : float
        The 'between' sample, or error mean sum of squares, usually denoted :math:`MSE`.
    f_statistic : float
        The computed :math:`F` statistic found by :math:`\frac{MST}{MSE}`.
    p_value : float
        The p-value from the :math:`F` distribution given the calculated :math:`F` statistic.
    analysis_type : str
        Name of the analysis performed, currently only reutns 'One-Way ANOVA'

    Notes
    -----
    One-way ANOVA can be considered an extension of the t-test when more than two groups
    are being tested. The factor, or categorical variable, is often referred to as the
    'treatment' in the ANOVA setting. ANOVA involves partitioning the data's total
    variation into variation between and within groups. This procedure is thus known as
    Analysis of Variance as sources of variation are examined separately.

    The data is assumed to be normally distributed with mean :math:`\mu_i` and standard
    deviation :math:`\sigma^2_i`. Stating the hypothesis is also similar to previous
    examples when there were only two samples of interest. The hypothesis can be defined
    formally as:

    :math:`H_O: \mu_1 = \mu_2 = \cdots = \mu_k`
    :math:`H_A:` Not all population means are equal

    The one-way ANOVA splits the data's variation into two sources which are in turn used
    to calculate the F-statistic. The F-statistic is determined by the F-test, which is
    done by dividing the variance between groups by the variance within groups. The sum of
    squares for treatments is defined as :math:`SST`, for error as :math:`SSE` and the total
    :math:`TotalSS`. The mean squares are calculated by dividing the sum of squares by the
    degrees of freedom.

    Each sum of squares can be defined as:

    .. math::

        SST = \sum_{i=1}^k n_i(\bar{y_{i}} - \bar{y})^2

    .. math::

        SSE = \sum_{i=1}^k (n_i - 1)s_i^2

    .. math::

        TotalSS = \sum_{i=1}^k \sum_{j=1}^{n_i} (y_{ij} - \bar{y})^2

    The mean squares are the sum of squares divided by the degrees of freedom.

    .. math::

        MST = \frac{SST}{k - 1}

    .. math::

        MSE = \frac{SSE}{n - k}

    The F-statistic is defined as:

    .. math::

        f = \frac{MST}{MSE}

    Examples
    --------
    There are several ways to perform a one-way ANOVA with the :code:`one_way_anova` function.
    Perhaps the simplest approach is to pass a group vector with the :code:`group` parameter
    and the corresponding observation vector as below.

    The data used in this example is a subset of the data obtained from the plant growth
    dataset given in Dobson (1983).

    >>> group_vector = ['ctrl', 'ctrl', 'ctrl',
    ...                 'trt1', 'trt1', 'trt1',
    ...                 'trt2', 'trt2', 'trt2']
    >>> observation_vec = [4.17, 5.58, 5.18,
    ...                    4.81, 4.17, 4.41,
    ...                    5.31, 5.12, 5.54]
    >>> aov = AnovaOneWay(observation_vec, group=group_vector)
    >>> aov.test_summary
    {'F-statistic': 2.4895587076438104,
     'Group DoF': 2,
     'Group Mean Squares': 0.5616444444444436,
     'Group Sum of Squares': 1.1232888888888872,
     'Group statistics': {'Group Means': [('ctrl', 4.976666666666667),
       ('trt1', 4.463333333333334),
       ('trt2', 5.323333333333333)],
      'Group Observations': [('ctrl', 3), ('trt1', 3), ('trt2', 3)],
      'Group Variance': [('ctrl', 0.5280333333333334),
       ('trt1', 0.10453333333333321),
       ('trt2', 0.04423333333333332)]},
     'Residual DoF': 6,
     'Residual Mean Squares': 0.2256,
     'Residual Sum of Squares': 1.3536,
     'Test description': 'One-Way ANOVA',
     'p-value': 0.163211765340447}

    The other approach is to pass each group sample vector similar to the below.

    >>> ctrl = [4.17, 5.58, 5.18]
    >>> trt1 = [4.81, 4.17, 4.41]
    >>> trt2 = [5.31, 5.12, 5.54]
    >>> aov1 = AnovaOneWay(ctrl, trt1, trt2)
    >>> aov1.test_summary
    {'F-statistic': 2.4895587076438104,
     'Group DoF': 2,
     'Group Mean Squares': 0.5616444444444436,
     'Group Sum of Squares': 1.1232888888888872,
     'Group statistics': {'Group Means': [('ctrl', 4.976666666666667),
       ('trt1', 4.463333333333334),
       ('trt2', 5.323333333333333)],
      'Group Observations': [('ctrl', 3), ('trt1', 3), ('trt2', 3)],
      'Group Variance': [('ctrl', 0.5280333333333334),
       ('trt1', 0.10453333333333321),
       ('trt2', 0.04423333333333332)]},
     'Residual DoF': 6,
     'Residual Mean Squares': 0.2256,
     'Residual Sum of Squares': 1.3536,
     'Test description': 'One-Way ANOVA',
     'p-value': 0.163211765340447}

    References
    ----------
    Dobson, A. J. (1983) An Introduction to Statistical Modelling.
        London: Chapman and Hall.

    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    """
    def __init__(self, *args, group=None):

        self.design_matrix = _build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.group_stats = self._group_statistics()
        self.group_names = np.unique(self.group)
        self.k = len(self.group_names)
        self.group_degrees_of_freedom = self.k - 1
        self.residual_degrees_of_freedom = len(self.design_matrix) - self.k

        self.group_sum_squares = self._sst()
        self.residual_sum_squares = self._sse()

        self.group_mean_squares = self._mst()
        self.residual_mean_squares = self._mse()

        self.f_statistic = self._fvalue()
        self.p_value = self._pvalue()
        self.analysis_type = 'One-Way ANOVA'
        self.test_summary = {
            'Analysis Performed': self.analysis_type,
            'F-statistic': self.f_statistic,
            'p-value': self.p_value,
            'Group DoF': self.group_degrees_of_freedom,
            'Residual DoF': self.residual_degrees_of_freedom,
            'Group Sum of Squares': self.group_sum_squares,
            'Group Mean Squares': self.group_mean_squares,
            'Residual Sum of Squares': self.residual_sum_squares,
            'Residual Mean Squares': self.residual_mean_squares,
            'Group Means': self.group_stats['Group Means'],
            'Group Obs Number': self.group_stats['Group Observations'],
            'Group Variance': self.group_stats['Group Variance']
        }

    def _sse(self):
        r"""
        Method for computing the 'within' sample sum of squares, also known as the sum of squares of the error
        partition.

        Returns
        -------
        sse : float
            The SSE of the design matrix

        Notes
        -----
        SSE is defined as the 'within' sample sum of squares and is computed as the sum of the number of
        observations in each group minus one, multiplied by the variance of the group sample. More formally,
        the 'within' sum of squares can be defined as:

        .. math::

            SSE = \sum_{i=1}^k (n_i - 1)s_i^2

        SSE is also known as the sum of squares of the 'error' partition of the total sum of squares.

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        group_n = np.array([i for _, i in self.group_stats['Group Observations']])
        group_variance = np.array([i for _, i in self.group_stats['Group Variance']])

        sse = np.sum((group_n - 1) * group_variance)

        return sse

    def _sst(self):
        r"""
        Computes the 'within' sample sum of squares. The 'within' sample sum of squares is also known as the
        sum of squares of the treatment partition.

        Returns
        -------
        sst : float
            The treatment sum of squares.

        Notes
        -----
        The 'between' sample sum of squares, also denoted SSH in some literature, is defined as:

        .. math::

            SST = \sum_{i=1}^k n_i(\bar{y_{i}} - \bar{y})^2

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        group_n = np.array([i for _, i in self.group_stats['Group Observations']])
        group_means = np.array([i for _, i in self.group_stats['Group Means']])
        total_mean = np.mean(self.design_matrix[:, 1])

        sst = np.sum(group_n * (group_means - total_mean) ** 2)

        return sst

    def _mst(self):
        r"""
        Computes the mean sum of squares of the treatment partition of the ANOVA design.

        Returns
        -------
        mst : float
            The mean treatment sum of squares.

        Notes
        -----
        The mean treatment sum of squares, also denoted MSH in some literature, is defined as the treatment
        sum of squares divided by the group degrees of freedom, :math:`k - 1`, where :math:`k` is the number of
        treatment groups. More formally, the MST can be written as:

        .. math::

            \frac{SST}{k - 1} = \frac{\sum_{i=1}^k n_i(\bar{y_{i}} - \bar{y})^2}{k - 1}

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        mst = self.group_sum_squares / self.group_degrees_of_freedom

        return mst

    def _mse(self):
        r"""
        Computes the mean sum of squares of the error partition.

        Returns
        -------
        mse : float
            The computed mean error sum of squares.

        Notes
        -----
        The mean 'within' sample sum of squares is defined as the error sum of squares partition divided by the
        residual degrees of freedom, :math:`n - k`, where :math`n` is the number of sample observations and
        :math:`k` is the number of treatment groups.

        .. math::

            \frac{SSE}{n - k} = \frac{\sum_{i=1}^k (n_i - 1)s_i^2}{n - k}

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        mse = self.residual_sum_squares / self.residual_degrees_of_freedom

        return mse

    def _fvalue(self):
        r"""
        Computes the analysis of variance F-statistic.

        Returns
        -------
        fval : float
            The corresponding F-statistic of the test.

        Notes
        -----
        The F-statistic is found by dividing the mean treatment sum of squares by the mean error sum of squares.

        .. math::

            \frac{MST}{MSE} = \frac{(\sum_{i=1}^k n_i(\bar{y_{i}} - \bar{y})^2)/(k - 1)}{(\sum_{i=1}^k (n_i - 1)s_i^2)/(n - k)}

        The F-statistic is distributed as :math:`F_{k-1, n-k}` when the null hypothesis :math:`H_0` is true. The
        null hypothesis is rejected if :math:`F > F_{\alpha}.

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        fval = self.group_mean_squares / self.residual_mean_squares

        return fval

    def _pvalue(self):
        r"""
        Returns the p-value using the computed F-statistic

        Returns
        -------
        p : float
            The computed p-value given the found F-statistic and the group and residual degrees of freedom.

        Notes
        -----
        The :code:`cdf` method from Scipy's :code:`stats.f` class is used to find the p-value.

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        p = 1 - f.cdf(self.f_statistic,
                      self.group_degrees_of_freedom,
                      self.residual_degrees_of_freedom)

        return p

    def _group_statistics(self):
        r"""
        Computes group summary statistics (mean, number of observations, and variance), for use when
        performing analysis of variance.

        Returns
        -------
        group_stats : dict
            Dictionary containing each group's mean, number of observations and variance.

        """
        group_means = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], np.mean)
        group_obs = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], len)
        group_variance = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], var)

        group_stats = {
            'Group Means': group_means,
            'Group Observations': group_obs,
            'Group Variance': group_variance
        }

        return group_stats


class BartlettsTest(object):
    r"""
    Performs Bartlett's Test for Homogenity of Variances of two or more sample groups.

    Parameters
    ----------
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector.
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector(s).

    Attributes
    ----------
    design_matrix : array-like
        Numpy ndarray representing the data matrix for the analysis.
    n : int
        Total number of samples in all groups.
    k : int
        Number of groups.
    test_statistic : float
        The computed Levene's Test statistic, typically denoted :math:`\chi^2`.
    p_value : float
        The associated p-value of the :math:`\chi^2` test statistic.
    test_description : str
        Description of the type of test performed.
    test_summary : dict
        Dictionary containing test results.

    Notes
    -----
    Bartlett's test, similar to Levene's Test and the Brown-Forsythe Test, is another procedure for determining if
    two or more sample groups have equal variances. These tests are commonly referred to as 'tests for homogenity of
    variance'. Bartlett's test is known to be sensitive when the samples are not normally distributed as the test uses
    mean square of the groups' deviations, also known as the pooled variance. Levene's Test and the Brown-Forsythe test
    are therefore alternatives to Bartlett's Test that are more performant when samples may depart from normality.

    The test statistic is approximately chi-square distributed with :math:`k - 1` degrees of freedom, where :math:`k`
    is the number of sample groups. The chi-square approximation does not hold sufficiently when the sample size of
    a group is :math:`n_i > 5`.

    The test statistic, :math:`\chi^2` is defined as:

    .. math::

        \chi^2 = \frac{(n - k) \ln(S^2_p) - \sum^k_{i=1} (n_i - 1) \ln(S^2_i)}{1 + \frac{1}{3(k - 1)} \left(\sum^k_{i=1} (\frac{1}{n_i - 1}) - \frac{1}{n - k} \right)}

    where :math:`n` is the total number of samples across all groups, :math:`k` is the number of groups, :math:`S^2_i`
    are the sample variances.

    :math:`S^2_p`, the pooled estimate of the samples' variance, is defined as:

    .. math::

        S^2_p = \frac{1}{n - k} \sum_i (n_i - 1) S^2_i

    Examples
    --------

    References
    ----------
    NIST/SEMATECH e-Handbook of Statistical Methods. Available online, URL:
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda357.htm

    Snedecor, George W. and Cochran, William G. (1989), Statistical Methods, Eighth Edition,
        Iowa State University Press.

    Wikipedia contributors. "Bartlett's test." Wikipedia, The Free Encyclopedia.
        Wikipedia, The Free Encyclopedia, 17 Feb. 2020. Web. 13 Mar. 2020.

    """
    def __init__(self, *args, group):

        self.design_matrix = _build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.n = self.design_matrix.shape[0]
        self.k = len(np.unique(self.design_matrix[:, 0]))
        self.test_description = "Bartlett's Test for Homogenity of Variances"
        self.test_statistic = self._bartlett()
        self.p_value = self._pvalue()
        self.test_summary = {
            'test_description': self.test_description,
            'test_statistic': self.test_statistic,
            'p_value': self.p_value
        }

    def _bartlett(self):
        group_n = np.array([i for _, i in npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], len)])
        group_variance = np.array([i for _, i in npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], var)])

        pool_var = 1 / (self.n - self.k) * np.sum((group_n - 1) * group_variance)

        x2_num = (self.n - self.k) * np.log(pool_var) - np.sum((group_n - 1) * np.log(group_variance))
        x2_den = 1 + 1 / (3 * (self.k - 1)) * (np.sum(1 / (group_n - 1)) - 1 / (self.n - self.k))

        x2 = x2_num / x2_den

        return x2

    def _pvalue(self):
        p = 1 - chi2.cdf(self.test_statistic, self.k - 1)

        return p


class LevenesTest(object):
    r"""
    Performs Levene's Test for Homogenity of Variances of two or more sample groups.

    Parameters
    ----------
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector.
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector(s).
    location : str, {'median', 'mean'}
        Specifies the procedure used to calculate Levene's Test. The default 'median', performs the standard Levene's
        Test while 'mean' will perform the Brown-Forsythe test for equality of variances.

    Attributes
    ----------
    design_matrix : array-like
        Numpy ndarray representing the data matrix for the analysis.
    n : int
        Total number of samples in all groups.
    k : int
        Number of groups.
    location : str
        The location parameter specifying which procedure to perform.
    test_statistic : float
        The computed Levene's Test statistic, typically denoted :math:`W`.
    p_value : float
        The associated p-value of the :math:`W` test statistic.
    test_description : str
        Description of the type of test performed.
    test_summary : dict
        Dictionary containing test results.

    Raises
    ------
    ValueError
        Raised if :code:`location` parameter is not one of 'median' or 'mean'.

    Notes
    -----
    The test statistic, :math:`W` used in Levene's test is defined as:

    .. math::

        W = \frac{(N - k)}{(k - 1)} \frac{\sum^k_{i=1} n_i (Z_{i.} - Z_{..})^2}{\sum^k_{i=1} \sum^{n_i}_{j=1} (Z_{ij} - Z_{i.})^2}

    where,

    - :math: `k` is the number of groups
    - :math: `n_i` is the number of samples belonging to the i-th group.
    - :math: `N` is the total number of samples.
    - :math:`Y_{ij}` is the jth observation from the ith group.

    and,

    .. math::

        Z_{i.} = \frac{1}{n_i} \sum^{n_i}_{j=1} Z_{ij}

        Z_{..} = \frac{1}{N} \sum^k_{i=1} \sum^{n_i}_{j=1} Z_{ij}

    are the mean of the calculated :math:`Z_{ij}` for group i and mean of all :math:`Z_{ij}`, respectively.

    In Levene's Test, :math:`Z_{ij}` is:

    .. math::

        |Y_{ij} - \tilde{Y}_i|

    where :math:`\tilde{Y}_i` is the median of the ith group.

    In the case of the Brown-Forsythe test, :math:`Z_{ij}` is:

    .. math::

        |Y_{ij} - \bar{Y}_i|

    where :math:`\bar{Y}_i` is the mean of the ith group.

    The :math:`W` test statistic is approximately :math:`F`-distributed with :math:`k - 1` and :math:`n - k` degrees
    of freedom, :math:`F(\alpha, k-1, n-k)`

    Examples
    --------

    References
    ----------
    NIST/SEMATECH e-Handbook of Statistical Methods. Available online, URL:
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm

    Snedecor, George W. and Cochran, William G. (1989), Statistical Methods, Eighth Edition,
        Iowa State University Press.

    Wikipedia contributors. "Levene's test." Wikipedia, The Free Encyclopedia.
        Wikipedia, The Free Encyclopedia, 28 Apr. 2019. Web. 13 Mar. 2020.

    """
    def __init__(self, *args, group, location='median'):

        if location not in ('median', 'mean'):
            raise ValueError('location parameter must be one of "median" (default), or "mean".')
        self.design_matrix = _build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.n = self.design_matrix.shape[0]
        self.k = len(np.unique(self.design_matrix[:, 0]))
        self.location = location
        self.test_statistic = self._levenes_test()
        self.p_value = self._p_value()

        if location == 'median':
            self.test_description = "Levene's Test for Homogenity of Variances"
        elif location == 'mean':
            self.test_description = "Brown-Forsythe for Homogenity of Variances"

        self.test_summary = {
            'test_description': self.test_description,
            'test_statistic (w)': self.test_statistic,
            'p_value': self.p_value,
            'location': self.location
        }

    def _levenes_test(self):
        r"""
        Performs Levene's Test or the Brown-Forsythe test (depending on location parameter).

        Returns
        -------
        w : float
            The computed test statistic

        """
        group_obs = np.array([i for _, i in npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], len)])

        group_locs = []
        if self.location == 'median':
            for i in self.design_matrix[:, 0]:
                group_locs.append(np.median(self.design_matrix[np.where(self.design_matrix[:, 0] == i)][:, 1]))
        elif self.location == 'mean':
            for i in self.design_matrix[:, 0]:
                group_locs.append(np.mean(self.design_matrix[np.where(self.design_matrix[:, 0] == i)][:, 1]))

        group_average_mat = np.column_stack([self.design_matrix, np.array(group_locs)])

        zij = np.abs(np.array(group_average_mat[:, 1] - group_average_mat[:, 2]))
        zij_mat = np.column_stack([group_average_mat, zij])

        zij_group_means = []
        for i in zij_mat[:, 0]:
            zij_group_means.append(np.mean(zij_mat[np.where(zij_mat[:, 0] == i)][:, 3]))

        zij_mat = np.column_stack([zij_mat, np.array(zij_group_means)])

        zij_group_means = np.array([i for _, i in npi.group_by(zij_mat[:, 0], zij_mat[:, 3], np.mean)])

        total_mean = np.mean(zij_mat[:, 3])

        num = np.sum(np.array(group_obs) * (zij_group_means - total_mean) ** 2)

        den = np.sum((zij_mat[:, 3] - zij_mat[:, 4]) ** 2)

        w = (self.n - self.k) / (self.k - 1) * (num / den)

        return w

    def _p_value(self):
        r"""
        Finds the associated p-value of the W test statistic.

        Returns
        -------
        p : float
            The computed p-value of the associated W test statistic.

        """
        p = 1 - f.cdf(self.test_statistic,
                      self.k - 1,
                      self.n - self.k)

        return p


class ManovaOneWay(object):
    r"""
    Performs multivariate analysis of variance, also known as MANOVA. Multivariate analysis of variance is
    the extension of the ANOVA procedure for two or more dependent variables.

    Parameters
    ----------
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector.
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector. If only one sample vector is passed with a group
        variable, one-way MANOVA will be performed.

    Attributes
    ----------
    design_matrix : array-like
        Numpy ndarray representing the data matrix for the analysis.
    group_names: array-like
        Numpy array of the group names.
    k : int
        The number of groups.
    group_stats : dict
        Dictionary containing group means, number of observations and data for each
        dependent variable.
    observation_stats : dict
        Dictionary of overall dependent variable means and number of observations.
    hypothesis_matrix : array-like
        The calculated hypothesis matrix, :math:`H`.
    error_matrix : array-like
        The error matrix, :math:`E`.
    degrees_of_freedom : dict
        Dictionary containing the group and residual degrees of freedom.
    numerator_dof : int
        The numerator (group) degrees of freedom.
    denominator_dof : int
        The denominator (residual) degrees of freedom.
    pillai_statistic : dict
        Dictionary containing the Pillai statistic, and the corresponding F-statistic and p-value.
    wilks_lambda : dict
        Dictionary containing the Wilk's Lambda statistic, and the corresponding F-statistic and p-value.
    roys_statistic : dict
        Dictionary containing Roy's statistic, and the corresponding F-statistic and p-value.
    hotelling_t2_statistic : dict
        Dictionary containing the Lawey-Hotelling :math:`T^2` statistic, and the corresponding
        F-statistic and p-value.
    analysis_type : str
        String denoting the type of analysis performed. Currently only returns 'One-Way MANOVA'

    Notes
    -----
    MANOVA, or Multiple Analysis of Variance, is an extension of Analysis of
    Variance (ANOVA) to several dependent variables. The approach to MANOVA
    is similar to ANOVA in many regards and requires the same assumptions
    (normally distributed dependent variables with equal covariance matrices).

    In the MANOVA setting, each observation vector can have a model denoted as:

    .. math::

        y_{ij} = \mu_i + \epsilon_{ij} \qquad i = 1, 2, \cdots, k; \qquad j = 1, 2, \cdots, n

    An 'observation vector' is a set of observations measured over several variables.
    With :math:`p` variables, :math:`y_{ij}` becomes:

    .. math::

        \begin{bmatrix} y_{ij1} \\ y_{ij2} \\ \vdots \\ y_{ijp} \end{bmatrix} = \begin{bmatrix}
        \mu_{i1} \\ \mu_{i2} \\ \vdots \\ \mu_{ip} \end{bmatrix} + \begin{bmatrix} \epsilon_{ij1}
        \\ \epsilon_{ij2} \\ \vdots \\ \epsilon_{ijp} \end{bmatrix}

    As before in ANOVA, the goal is to compare the groups to see if there are any significant
    differences. However, instead of a single variable, the comparisons will be made with the
    mean vectors of the samples. The null hypothesis :math:`H_0` can be formalized the same
    way in MANOVA:

    .. math::

        H_0: \mu_1 = \mu_2 = \dots = \mu_k

    With an alternative hypothesis :math:`H_a` that at least two :math:`\mu` are unequal.
    There are :math:`p(k - 1)`, where :math:`k` is the number of groups in the data,
    equalities that must be true for :math:`H_0` to be accepted.

    Similar to ANOVA, we are interested in partitioning the data's total variation into
    variation between and within groups. In the case of ANOVA, this partitioning is done
    by calculating :math:`SSH` and :math:`SSE`; however, in the multivariate case, we must
    extend this to encompass the variation in all the :math:`p` variables. Therefore, we
    must compute the between and within sum of squares for each possible comparison. This
    procedure results in the :math:`H` "hypothesis matrix" and :math:`E` "error matrix."

    The :math:`H` matrix is a square :math:`p \times p` with the form:

    .. math::

        H = \begin{bmatrix} SSH_{11} & SPH_{21} & \dots & SPH_{1p} \\
        SPH_{12} & SSH_{22} & \dots & SPH_{2p} \\ \vdots & \vdots & & \vdots \\
        SPH_{1p} & SPH_{2p} & \cdots & SSH_{pp} \end{bmatrix}

    The error matrix :math:`E` is also :math:`p \times p`

    .. math::

        E = \begin{bmatrix} SSE_{11} & SPE_{12} & \cdots & SPE_{1p} \\
        SPE_{12} & SSE_{22} & \cdots & SPE_{2p} \\ \vdots & \vdots & & \vdots \\
        SPE_{1p} & SPE_{2p} & \cdots & SSE_{pp} \end{bmatrix}

    Once the :math:`H` and :math:`E` matrices are constructed, the mean vectors can be
    compared to determine if significant differences exist. There are several test
    statistics, of which the most common are Wilk's lambda, Roy's test, Pillai, and
    Lawley-Hotelling, that can be employed to test for significant differences. Each test
    statistic has specific properties and power.

    Examples
    --------
    The data used in this example is a subset of the rootstock dataset used in Rencher (n.d.).
    The rootstock data contains four dependent variables and a group variable described as follows:

    1. tree_number: group membership indicator column
    2. trunk_girth_four_years: trunk girth at four years (mm :math:`\times` 100)
    3. ext_growth_four_years: extension growth at four years (m)
    4. trunk_girth_fifteen_years: trunk girth at 15 years (mm :math:`\times` 100)
    5. weight_above_ground_fifteen_years: weight of tree above ground at 15 years (lb :math:`\times` 1000)

    >>> tree_number = [1, 1, 1,
    ...                2, 2, 2,
    ...                3, 3, 3,
    ...                4, 4, 4,
    ...                5, 5, 5,
    ...                6, 6, 6]
    >>> trunk_girth_four_years = [1.11, 1.19, 1.09,
    ...                           1.05, 1.17, 1.11,
    ...                           1.07, 0.99, 1.06,
    ...                           1.22, 1.03, 1.14,
    ...                           0.91, 1.15, 1.14,
    ...                           1.11, 0.75, 1.05]
    >>> ext_growth_four_years = [2.569, 2.928, 2.865,
    ...                          2.074, 2.885, 3.378,
    ...                          2.505, 2.315, 2.667,
    ...                          2.838, 2.351, 3.001,
    ...                          1.532, 2.552, 3.083,
    ...                          2.813, 0.840, 2.199]
    >>> trunk_girth_fifteen_years = [3.58, 3.75, 3.93,
    ...                              4.09, 4.87, 4.98,
    ...                              3.76, 4.44, 4.38,
    ...                              3.89, 4.05, 4.05,
    ...                              4.04, 4.16, 4.79,
    ...                              3.76, 3.14, 3.75]
    >>> weight_above_ground_fifteen_years = [0.760, 0.821, 0.928,
    ...                                      1.036, 1.094, 1.635,
    ...                                      0.912, 1.398, 1.197,
    ...                                      0.944, 1.241, 1.023,
    ...                                      1.084, 1.151, 1.381,
    ...                                      0.800, 0.606, 0.790]
    >>> maov = ManovaOneWay(trunk_girth_four_years, ext_growth_four_years,
    ...                       trunk_girth_fifteen_years, weight_above_ground_fifteen_years,
    ...                       group=tree_number)
    >>> maov.test_summary
    {'Dependent variable num.': 4,
     'Group Means': array([[1.13      , 2.78733333, 3.75333333, 0.83633333],
            [1.11      , 2.779     , 4.64666667, 1.255     ],
            [1.04      , 2.49566667, 4.19333333, 1.169     ],
            [1.13      , 2.73      , 3.99666667, 1.06933333],
            [1.06666667, 2.389     , 4.33      , 1.20533333],
            [0.97      , 1.95066667, 3.55      , 0.732     ]]),
     'Group Num. Observations': [3, 3, 3, 3, 3, 3],
     'Hotellings T^2': {'Hotellings T^2 F-value': 13.787005512065765,
      'Hotellings T^2 Statistic': 5.743438210016407,
      'Hotellings T^2 p-value': 0.0001270639867039236},
     'Observation Total Means': array([1.07444444, 2.52194444, 4.07833333, 1.0445    ]),
     'Observations': {'x means': array([1.07444444, 2.52194444, 4.07833333, 1.0445    ]),
      'x observations': 4},
     'Pillai Statistic': {'Pillai F-value': 1.5309502494809615,
      'Pillai Statistic': 1.557842406866489,
      'Pillai p-value': 0.2522352735968698},
     'Roys Statistic': {'Roys Statistic': 4.595030059073131,
      'Roys Statistic F-value': 93.73861320509187,
      'Roys Statistic p-value': 3.4357116041050517e-09},
     'Test Description': 'One-Way MANOVA',
     'Wilks Lambda': {'Wilks Lambda': 0.07218625211663433,
      'Wilks Lambda F-value': 1.861776394897166,
      'Wilks Lambda p-value': 0.17516209487139456},
     'degrees of freedom': {'Denominator Degrees of Freedom': 12,
      'Numerator Degrees of Freedom': 5.0}}

    References
    ----------
    Andrews, D. F., and Herzberg, A. M. (1985), Data, New York: Springer-Verlag.

    Fox J. and Weisberg, S. (2011) An R Companion to Applied Regression, Second Edition Sage.

    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    """
    def __init__(self, *args, group):

        self.design_matrix = _build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.k = len(np.unique(self.group))

        self.group_stats = self._group_statistics()
        self.observation_stats = self._obs_statistics()

        self.hypothesis_matrix, self.error_matrix = self._hypothesis_error_matrix()
        self._intermediate_statistic_parameters = self._intermediate_test_statistic_parameters()

        self.degrees_of_freedom = self._degrees_of_freedom()
        self.numerator_dof = self.degrees_of_freedom['Numerator Degrees of Freedom']
        self.denominator_dof = self.degrees_of_freedom['Denominator Degrees of Freedom']
        self.pillai_statistic = self._pillai_statistic()
        self.wilks_lambda = self._wilks_statistic()
        self.roys_statistic = self._roys_statistic()
        self.hotelling_t2_statistic = self._hotelling_t2_statistic()
        self.analysis_type = 'One-Way MANOVA'
        self.test_summary = self._generate_result_summary()

    def _hypothesis_error_matrix(self):
        r"""
        Computes the 'hypothesis' matrix, :math:`H` and the 'error' matrix, :math:`E`.

        Returns
        -------
        h, e : array-like
            The :math:`k \times k` 'hypothesis' and 'error' matrix of the MANOVA design.

        Notes
        -----
        The hypothesis, :math:`H`, and error :math:`E` matrices correspond to the treatment and
        error sum of squares that are calculated when performing univariate analysis of variance.
        :math:`H` is defined as:

        .. math::

            H = n \sum_{i=1}^k (\bar{y}_{i.} - \bar{y}_{..}) (\bar{y}_{i.} - \bar{y}_{..})^{prime}

        and has the form:

        .. math::

            H = \begin{bmatrix}
                    SSH_{11} & SPH_{12} & \ldots & SPH_{1p} \\
                    SPH_{12} & SSH_{22} & \ldots & SPH_{2p} \\
                    \vdots & \vdots & \vdots \\
                    SPH_{1p} & SPH_{2p} & \ldots & SSH_{pp}
                \end{bmatrix}

        Where,

        .. math::

            SSH_{11} = n \sum_{i=1}^k (\bar{y}_{i.1} - \bar{y}_{..1})^2 =
            \sum_i \frac{y_{i.1}^2}{n} - \frac{y_{..1}^2}{kn}

            SPH_{23} = n \sum_{i=1}^k (\bar{y}_{i.2} - \bar{y}_{..2}) (\bar{y}_{i.3} - \bar{y}_{..3}} =
            \sum_i \frac{y_{i.2} y_{i.3}}{n} - \frac{y_{..2} y_{..3}}{kn}

        The diagonal elements of the :math:`H` matrix represent the between sum of squares for each of the
        dependent variables, while the off-diagonal elements are sums of products for each pair of variables.
        The rank of :math:`H` is the smaller of the number of dependent variables, :math:`k` and :math:`v_H`,
        where :math:`v_H` is the degrees of freedom, :math:`k - 1`.

        The error, :math:`E` matrix can also be expressed similarly to the form of the hypothesis matrix.
        For example,

        .. math::

            E = \begin{matrix}
                    SSE_{11} & SPE_{12} & \ldots & SPE_{1p} \\
                    SPE_{12} & SSE_{22} & \ldots & SPE_{2p} \\
                    \vdots & \vdots & & \vdots \\
                    SPE_{1p} & SPE_{2p} & \ldots & SSE_{pp}
                \end{bmatrix}

        Where, for example, the elements of the matrix can be written as:

        .. math::

            SSE_{11} = \sum_{i=1}^k \sum_{j=1}^n (y_{ij1} - \bar{y}_{i.1})^2 =
            \sum_{ij} y_{ij1}^2 - \sum_{i} \frac{y_{i.1}^2}{n}

            SPE_{23} = \sum_{i=1}^k \sum_{j=1}^n (y_{ij2} - \bar{y}_{i.2}) (y_{ij3} - \bar{y}_{i.3}) =
            \sum_{ij} y_{ij2} y_{ij3} - sum_{i} \frac{y_{i.2} y_{i.3}}{n}

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        groupmeans = self.group_stats['Group Means']
        xmeans = self.observation_stats['x means']

        n = self.group_stats['Group Observations']
        xn = len(xmeans)

        h, e = np.zeros((xn, xn)), np.zeros((xn, xn))

        for i in np.arange(xn):
            for j in np.arange(i + 1):

                h[i, j] = n[i] * np.sum((groupmeans[:, i] - xmeans[i]) * (groupmeans[:, j] - xmeans[j]))
                h[j, i] = n[i] * np.sum((groupmeans[:, j] - xmeans[j]) * (groupmeans[:, i] - xmeans[i]))

                b = []

                for k in self.group_stats['Groups']:
                    a = np.sum((k[:, i] - np.mean(k[:, i])) * (k[:, j] - np.mean(k[:, j])))
                    b.append(a)

                e[i, j], e[j, i] = np.sum(b), np.sum(b)

        return h, e

    def _pillai_statistic(self):
        r"""
        Computes the Pillai statistic, a commonly used statistic of significance when performing multivariate
        analysis of variance.

        Returns
        -------
        pillai_stat : dict
            Dictionary containing the Pillai statistic, and the corresponding F-statistic and p-value.

        Notes
        -----
        The Pillai test statistic is denoted as :math:`V^{(s)}` and defined as:

        .. math::

            V^{(s)} = tr[(E + H0)^{-1} H] = \sum^s_{i=1} \frac{\lambda_i}{1 + \lambda_i}

        Where :math:`\lambda_i` represents the :math:`i`th nonzero eigenvalue of
        :math:`E^{-1}H`.

        The critical Pillai value is found by computing :math:`s`, :math:`m`, and :math:`N`
        which are also employed in Roy's test (the Pillai test is an extension of Roy's
        test). The values are defined as:

        .. math::

            s = min(p, V_h) \qquad m = \frac{1}{2} (\left| V_h - p \right| - 1) \qquad N = \frac{1}{2} (V_E - p - 1)

        An approximate F-statistic can be found with the following equation:

        .. math::

            F = \frac{(2N + s + 1)V^{(s)}}{(2m + s + 1)(s - V^{(s)})}

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        nn, s, m = self._intermediate_statistic_parameters['nn'], self._intermediate_statistic_parameters['s'], \
                   self._intermediate_statistic_parameters['m']

        pillai = np.sum(np.diag(np.dot(np.linalg.inv(self.hypothesis_matrix + self.error_matrix),
                                       self.hypothesis_matrix)))

        pillai_f = ((2. * nn + s + 1.) * pillai) / ((2. * m + s + 1.) * (s - pillai))

        pillai_stat = {'Pillai Statistic': pillai,
                       'Pillai F-value': pillai_f,
                       'Pillai p-value': self._p_value(pillai_f, self.numerator_dof, self.denominator_dof)}

        return pillai_stat

    def _wilks_statistic(self):
        r"""
        Wilk's lambda is another commonly employed test of significance used in multivariate analysis of
        variance

        Returns
        -------
        wilks_stat : dict
            Dictionary containing the Wilk's lambda statistic, and the corresponding F-statistic and p-value.

        Notes
        -----
        Wilk's :math:`\Lambda`, one of the more commonly used MANOVA test statistics,
        compares the :math:`E` matrix to the total :math:`E + H` matrix. Wilk's
        :math:`\Lambda` is calculated using the determinants of those two matrices.

        .. math::

            \Lambda = \frac{\left| E \right|}{\left| E + H \right|}

        Wilk's :math:`\Lambda` can also be written in terms of the :math:`E^{-1}H`
        eigenvalues :math:`(\lambda_1, \lambda_2, \cdots, \lambda_s)`.

        .. math::

            \Lambda = \prod^s_{i=1} \frac{1}{1 + \lambda_i}

        An approximate F-Value can be calculated using Wilk's Lambda:

        .. math::

            F = \frac{1 - \Lambda^{(1/t)}}{\Lambda^{(1/t)}} \frac{df_2}{df_1}

        Where

        .. math::

            df_1 = pv_H, \qquad df_2 = wt - \frac{1}{2}(pv_H - 2), \qquad w = v_E + v_H - \frac{1}{2}(p + v_H + 1)

        .. math::

            t = \sqrt{\frac{p^2v^2_H - 4}{p^2 + v^2_H - 5}}

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        vh, p, ve, eigs = self._intermediate_statistic_parameters['vh'], \
                          self._intermediate_statistic_parameters['p'], \
                          self._intermediate_statistic_parameters['ve'], \
                          self._intermediate_statistic_parameters['eigs']

        t = np.sqrt((p ** 2. * vh ** 2. - 4.) / (p ** 2. + vh ** 2. - 5.))
        df1 = p * vh
        df2 = (ve + vh - .5 * (p + vh + 1.)) * t - .5 * (p * vh - 2.)

        wilks_lambda = np.prod(1. / (1. + eigs))
        wilks_lambda_f = ((1. - wilks_lambda ** (1. / t)) / wilks_lambda ** (1. / t)) * (df2 / df1)

        wilks_stat = {"Wilks Lambda": wilks_lambda,
                      "Wilks Lambda F-value": wilks_lambda_f,
                      "Wilks Lambda p-value": self._p_value(wilks_lambda_f, self.numerator_dof, self.denominator_dof)}

        return wilks_stat

    def _roys_statistic(self):
        r"""
        Roy's test statistic, also known as Roy's largest root test, is another test of significance used in
        multivariate analysis of variance.

        Returns
        -------
        roy_stat : dict
            Dictionary containing the Roy's statistic value, and the corresponding F-statistic and p-value.

        Notes
        -----
        Roy's test statistic is the largest eigenvalue of the matrix :math:`E^{-1}H`

        The F-statistic in the Roy's statistic setting can be computed as:

        .. math::

            \frac{k(n - 1)}{k - 1} \lambda_1

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        eigs, n, vh = self._intermediate_statistic_parameters['eigs'], \
                      self._intermediate_statistic_parameters['n'], \
                      self._intermediate_statistic_parameters['vh']

        roy = np.max(eigs)
        roy_f = float(self.k * (n - 1)) / float(vh) * roy

        roy_stat = {"Roys Statistic": roy,
                    "Roys Statistic F-value": roy_f,
                    "Roys Statistic p-value": self._p_value(roy_f, self.numerator_dof, self.denominator_dof)}

        return roy_stat

    def _hotelling_t2_statistic(self):
        r"""
        Computes the Lawley-Hotelling :math:`T^2` test statistic, another test of significance used in
        multivariate analysis of variance.

        Returns
        -------
        t2_stat : dict
            Dictionary containing the calculated Lawley-Hotelling :math:`T^2` statistic, and the
            corresponding F-statistic and p-value.

        Notes
        -----
        The Lawley-Hotelling statistic, also known as Hotelling's generalized :math:`T^2`-statistic,
        is denoted :math:`U^{(s)}` and is defined as:

        .. math::

            U^{(s)} = tr(E^{-1}H) = \sum^s_{i=1} \lambda_i

        The approximate F-statistic in the Lawley-Hotelling setting is defined as:

        .. math::

            F = \frac{2 (sN + 1)U^{(s)}}{s^2(2m + s + 1)}

        Where :math:`s`, :math:`m`, and :math`N` are defined as above.

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        dot_inve_h, s, nn, m = self._intermediate_statistic_parameters['dot_inve_h'], \
                               self._intermediate_statistic_parameters['s'], \
                               self._intermediate_statistic_parameters['nn'], \
                               self._intermediate_statistic_parameters['m']

        t2 = np.sum(np.diag(dot_inve_h))
        t2_f = (2. * (s * nn + 1.) * np.sum(dot_inve_h)) / (s ** 2. * (2. * m + s + 1.))

        t2_stat = {
            "Hotellings T^2 Statistic": t2,
            "Hotellings T^2 F-value": t2_f,
            "Hotellings T^2 p-value": self._p_value(t2_f, self.numerator_dof, self.denominator_dof)
        }

        return t2_stat

    @staticmethod
    def _p_value(f_val, df_num, df_denom):
        r"""
        Returns the p-value using the computed F-statistic

        Parameters
        ----------
        f_val : float
            The calculated F-statistic found from test of significance procedures (Pillai, Wilk's Lambda, Roy's
            statistic, and Lawley-Hotelling :math:`T^2`).
        df_num : int
            The hypothesis degrees of freedom, :math:`k - 1`, where :math:`k` is the number of groups.
        df_denom : int
            The error degrees of freedom, :math:`n - k`, where :math:`n` is the number of total observations
            and :math:`k` is the number of groups.

        Returns
        -------
        p : float
            The computed p-value given the found F-statistic and the group and residual degrees of freedom.

        Notes
        -----
        The :code:`cdf` method from Scipy's :code:`stats.f` class is used to find the p-value.

        References
        ----------
        Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        p = 1 - f.cdf(f_val, df_num, df_denom)

        return p

    def _generate_result_summary(self):
        r"""
        Returns a summary of the fitted MANOVA model as a dictionary.

        Returns
        -------
        anova_results : dict
            A summary of the fitted MANOVA model and relevant test statistics.

        """
        group_stats = self.group_stats
        group_stats.pop('Groups')

        manova_result = {
            'Analysis Performed': self.analysis_type,
            'degrees of freedom': self.degrees_of_freedom,
            'Pillai Statistic': self.pillai_statistic,
            "Wilks Lambda": self.wilks_lambda,
            "Roys Statistic": self.roys_statistic,
            "Hotellings T^2": self.hotelling_t2_statistic,
            'Group Means': group_stats['Group Means'],
            'Group Num. Observations': group_stats['Group Observations'],
            'Observation Total Means': self.observation_stats['x means'],
            'Dependent variable num.': self.observation_stats['x observations'],
            'Observations': self.observation_stats
        }

        return manova_result

    def _group_statistics(self):
        group_means = npi.group_by(self.design_matrix[:, 0]).mean(self.design_matrix[:, 1:])[1]

        group_observations = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1:], len)
        group_observations = [i for _, i in group_observations]

        groups = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1:])[1]

        group_stats = {
            'Group Means': group_means,
            'Group Observations': group_observations,
            'Groups': groups
        }

        return group_stats

    def _obs_statistics(self):
        x_means = self.design_matrix[:, 1:].mean(axis=0)
        x_group_observations = len(x_means)

        obs_stats = {
            'x means': x_means,
            'x observations': x_group_observations
        }

        return obs_stats

    def _intermediate_test_statistic_parameters(self):

        dot_inve_h = self._dot_inve_h(self.hypothesis_matrix, self.error_matrix)
        eigs = np.linalg.eigvals(dot_inve_h)

        p = len(self.error_matrix)
        n = self.design_matrix.shape[0]

        vh = self.k - 1.
        ve = n - self.k

        s = np.minimum(vh, p)
        m = 0.5 * (np.absolute(vh - p) - 1)
        nn = 0.5 * (ve - p - 1)

        intermediate_statistic_parameters = {
            'dot_inve_h': dot_inve_h,
            'eigs': eigs,
            'p': p,
            'n': n,
            'vh': vh,
            've': ve,
            's': s,
            'm': m,
            'nn': nn
        }

        return intermediate_statistic_parameters

    def _degrees_of_freedom(self):
        vh, ve, xn = self._intermediate_statistic_parameters['vh'], \
                     self._intermediate_statistic_parameters['ve'], \
                     len(self.observation_stats['x means'])

        num_df, denom_df = vh, ve

        dof = {'Numerator Degrees of Freedom': num_df,
               'Denominator Degrees of Freedom': denom_df}

        return dof

    @staticmethod
    def _dot_inve_h(h, e):
        return np.dot(np.linalg.inv(e), h)
