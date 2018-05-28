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


class ManovaOneWay(object):

    def __init__(self, *args, group):

        self.design_matrix = build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]


        self.design_matrix = build_des_mat(group, *args)
        #self.group_names = np.unique(self.group)
        self.k = len(np.unique(self.group))

        self._group_stats = self._group_statistics()
        self._observation_stats = self._obs_statistics()

        self.hypothesis_matrix, self.error_matrix = self._hypothesis_error_matrix()
        self._intermediate_statistic_parameters = self._intermediate_test_statistic_parameters()

        #self.degrees_of_freedom = self._degrees_of_freedom()
        #self.pillai_statistic = self.pillai_statistic()
        #self.wilks_lambda = self._wilks_statistic()
        #self.roys_statistic = self._roys_statistic()
        #self.hotelling_t2_statistic = self._hotelling_t2_statistic()

    def _group_statistics(self):
        group_means = npi.group_by(self.design_matrix[:, 0]).mean(self.design_matrix[:, 1])[1] #.mean(self.design_matrix[:, 1:])[1][:, 1:]

        group_observations = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1:], len)
        group_observations = [i for _, i in group_observations]

        groups = npi.group_by(self.design_matrix[:, 0], self.design_matrix)[1]

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

    def _hypothesis_error_matrix(self):

        groupmeans = self._group_stats['Group Means']
        xmeans = self._observation_stats['x means']

        xn = self._observation_stats['x observations']
        n = self._group_stats['Group Observations']

        h, e = np.zeros((xn, xn)), np.zeros((xn, xn))

        for i in np.arange(xn):
            for j in np.arange(i + 1):

                h[i, j] = n[i] * np.sum((groupmeans[:, i] - xmeans[i]) * (groupmeans[:, j] - xmeans[j]))
                h[j, i] = n[i] * np.sum((groupmeans[:, j] - xmeans[j]) * (groupmeans[:, i] - xmeans[i]))

                b = []

                for k in self._group_stats['Groups']:
                    a = np.sum((k[:, i] - np.mean(k[:, i])) * (k[:, j] - np.mean(k[:, j])))
                    b.append(a)

                e[i, j], e[j, i] = np.sum(b), np.sum(b)

        return h, e

    def _intermediate_test_statistic_parameters(self):

        dot_inve_h = self._dot_inve_h(self.hypothesis_matrix, self.error_matrix)
        eigs = np.linalg.eigvals(dot_inve_h)

        p = len(self.error_matrix)
        n = len(self.x)

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

    def _pillai_statistic(self):

        nn, s, m = self._intermediate_statistic_parameters['nn'], \
                   self._intermediate_statistic_parameters['s'], \
                   self._intermediate_statistic_parameters['m']

        pillai = np.sum(np.diag(np.dot(np.linalg.inv(self.hypothesis_matrix + self.error_matrix),
                                       self.hypothesis_matrix)))

        pillai_f = ((2. * nn + s + 1.) * pillai) / ((2. * m + s + 1.) * (s - pillai))

        pillai_stat = {'Pillai Statistic': pillai,
                       'Pillai F-value': pillai_f}

        return pillai_stat

    def _wilks_statistic(self):
        vh, p, ve, eigs = self._intermediate_statistic_parameters['vh'], \
                          self._intermediate_statistic_parameters['p'], \
                          self._intermediate_statistic_parameters['ve'], \
                          self._intermediate_statistic_parameters['eigs']

        t = np.sqrt((p ** 2. * vh ** 2. - 4.) / (p ** 2. + vh ** 2. - 5.))
        df1 = p * vh
        df2 = (ve + vh - .5 * (p + vh + 1.)) * t - .5 * (p * vh - 2.)

        wilks_lambda = np.prod(1. / (1. + eigs))
        wilks_lambda_f = ((1. - wilks_lambda ** (1. / t)) / wilks_lambda ** (1. / t)) * (df2 / df1)

        wilks_stat = {"Wilk's Lambda": wilks_lambda,
                      "Wilk's Lambda F-value": wilks_lambda_f}

        return wilks_stat

    def _roys_statistic(self):
        eigs, n, vh = self._intermediate_statistic_parameters['eigs'], \
                      self._intermediate_statistic_parameters['n'], \
                      self._intermediate_statistic_parameters['vh']

        roy = np.max(eigs)
        roy_f = float(self.k * (n - 1)) / float(vh) * roy

        roy_stat = {"Roy's Statistic": roy,
                    "Roy F-value": roy_f}

        return roy_stat

    def _hotelling_t2_statistic(self):
        dot_inve_h, s, nn, m = self._intermediate_statistic_parameters['dot_inve_h'], \
                               self._intermediate_statistic_parameters['s'], \
                               self._intermediate_statistic_parameters['nn'], \
                               self._intermediate_statistic_parameters['m']

        t2 = np.sum(np.diag(dot_inve_h))
        t2_f = (2. * (s * nn + 1.) * np.sum(dot_inve_h)) / (s ** 2. * (2. * m + s + 1.))

        t2_stat = {
            "Hotelling's T^2 Statistic": t2,
            "Hotelling's T^2 F-value": t2_f
        }

        return t2_stat

    def _degrees_of_freedom(self):
        vh, ve, xn = self._intermediate_statistic_parameters['vh'], \
                     self._intermediate_statistic_parameters['ve'], \
                     self._intermediate_statistic_parameters['xn']

        num_df, denom_df = vh * xn, ve * xn

        dof = {'Numerator Degrees of Freedom': num_df,
               'Denominator Degrees of Freedom': denom_df}

        return dof

    def _f_p_value(self, f, df_num, df_denom):
        p = 1 - f.cdf(f, df_num, df_denom)

        return p

    @staticmethod
    def _dot_inve_h(h, e):
        return np.dot(np.linalg.inv(e), h)