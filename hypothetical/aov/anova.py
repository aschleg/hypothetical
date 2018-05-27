import numpy as np
import numpy_indexed as npi
from scipy.stats import f

from hypothetical._lib import build_des_mat


def anova_one_way(*args, group=None):

    if len(args) == 1:
        return AnovaOneWay(*args, group=group)
    else:
        pass


class AnovaOneWay(object):
    r"""
    Performs one-way analysis of variance (ANOVA) of one measurement and a grouping variable

    Parameters
    ----------
    group
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the x parameter.
    x
        One or two-dimensional array (Numpy ndarray, Pandas DataFrame, list of lists) that
        defines the observation vectors of the dependent variables. Must be the same length
        as the group parameter.

    Returns
    -------
    namedtuple
        Namedtuple with the following entries representing an ANOVA table:
        residual Df: Residuals Degrees of Freedom
        Group Df: Group Vector Degrees of Freedom
        F-Value: Computed F-Value of ANOVA procedure
        p-value: Resulting p-value
        Group Sum of Squares: SST
        Group Mean Squares: MST
        Residual Sum of Squares: SSE
        Residual Mean Squares: MSE

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

    References
    ----------
    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    """
    def __init__(self, *args, group):

        self.design_matrix = build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.group_statistics = self._group_statistics()
        self.group_names = np.unique(self.group)
        self.k = len(self.group_names)
        self.group_degrees_of_freedom = self.k - 1
        self.residual_degrees_of_freedom = len(self.design_matrix) - self.k
        self.sst = self._sst()
        self.sse = self._sse()
        self.mst = self._mst()
        self.mse = self._mse()
        self.residual_mean_squares = self.mse
        self.residual_sum_squares = self.sse
        self.group_sum_squares = self.sst
        self.group_mean_squares = self.mst
        self.f_statistic = self._fvalue()
        self.p_value = self._pvalue()
        self.test_description = 'One-Way ANOVA'

    def _sse(self):
        group_n = self.group_statistics['Group Observations']
        group_variance = self.group_statistics['Group Variance']

        sse = 0

        for i, j in zip(group_n, group_variance):
            sse += (i[1] - 1) * j[1]

        return sse

    def _sst(self):
        group_n = self.group_statistics['Group Observations']
        group_means = self.group_statistics['Group Means']
        total_mean = np.mean(self.design_matrix[:, 1:])

        sst = 0
        for i, j in zip(group_n, group_means):
            sst += i[1] * (j[1] - total_mean) ** 2

        return sst

    def _mst(self):
        mst = self.sst / self.group_degrees_of_freedom

        return mst

    def _mse(self):
        mse = self.sse / self.residual_degrees_of_freedom

        return mse

    def _fvalue(self):
        return self.mst / self.mse

    def _pvalue(self):
        p = 1 - f.cdf(self.f_statistic,
                      self.group_degrees_of_freedom,
                      self.residual_degrees_of_freedom)

        return p

    def _group_statistics(self):
        group_means = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], np.mean)
        group_obs = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], len)
        group_variance = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], np.var)

        group_stats = {
            'Group Means': group_means,
            'Group Observations': group_obs,
            'Group Variance': group_variance
        }

        return group_stats

    def summary(self):
        anova_results = {
            'F statistic': self.f_statistic,
            'p-value': self.p_value,
            'group DoF': self.group_degrees_of_freedom,
            'residual DoF': self.residual_degrees_of_freedom,
            'group Sum of Squares': self.group_sum_squares,
            'group Mean Squares': self.group_mean_squares,
            'residual Sum of Squares': self.residual_sum_squares,
            'residual Mean Squares': self.residual_mean_squares,
            'group statistics': self.group_statistics
        }

        return anova_results