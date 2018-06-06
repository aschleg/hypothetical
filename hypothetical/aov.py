# encoding=utf8

"""

One-Way Analysis of Variance
----------------------------

.. autosummary::
    :toctree: generated/

    anova_one_way
    manova_one_way

References
----------
Andrews, D. F., and Herzberg, A. M. (1985), Data, New York: Springer-Verlag.

Dobson, A. J. (1983) An Introduction to Statistical Modelling.
        London: Chapman and Hall.

Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

"""


import numpy as np
import numpy_indexed as npi
from scipy.stats import f

from hypothetical._lib import build_des_mat
from hypothetical.summary import var


def anova_one_way(*args, group=None):
    r"""
    Performs one-way ANOVA. One-way ANOVA (Analysis of Variance) is used to analyze and test
    the differences of two or more groups have the same population mean.

    Parameters
    ----------
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector.
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector. If more than one sample vector is passed and
        the group parameter is not None, one-way MANOVA will be performed.

    Returns
    -------
    AnovaOneWay or ManovaOneWay : class object
        ANOVA or MANOVA (if more than one observation vector is passed with a group variable) class object
        containing the fitted results.

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
    >>> aov = anova_one_way(observation_vec, group=group_vector)
    >>> aov.summary()

    The other approach is to pass each group sample vector similar to the below.

    >>> ctrl = [4.17, 5.58, 5.18]
    >>> trt1 = [4.81, 4.17, 4.41]
    >>> trt2 = [5.31, 5.12, 5.54]
    >>> aov1 = anova_one_way(ctrl, trt1, trt2)
    >>> aov1.summary()

    See Also
    --------
    AnovaOneWay : one-way ANOVA class
    ManovaOneWay : one-way MANOVA class

    References
    ----------
    Dobson, A. J. (1983) An Introduction to Statistical Modelling.
        London: Chapman and Hall.

    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    """
    if len(args) == 1:
        aov_result = AnovaOneWay(*args, group=group)
    else:
        aov_result = ManovaOneWay(*args, group=group)

    return aov_result


def manova_one_way(*args, group=None):
    r"""

    Parameters
    ----------
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector.
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector. If more than one sample vector is passed and
        the group parameter is not None, one-way MANOVA will be performed.

    Returns
    -------
    AnovaOneWay or ManovaOneWay : class object
        MANOVA or ANOVA (if only one observation vector is passed with a group variable) class object
        containing the fitted results.

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
    Similar to the :code:`anova_one_way` function, there are several approaches to performing
    multivariate analysis of variance with :code:`manova_one_way`.

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
    >>> trunk_girth_fifteen_years = []

    References
    ----------
    Andrews, D. F., and Herzberg, A. M. (1985), Data, New York: Springer-Verlag.

    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    """
    if len(args) > 1:
        maov_result = ManovaOneWay(*args, group=group)
    else:
        maov_result = AnovaOneWay(*args, group=group)

    return maov_result


class AnovaOneWay(object):
    r"""

    Parameters
    ----------
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector.
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector. If more than one sample vector is passed and
        the group parameter is not None, one-way MANOVA will be performed.

    Attributes
    ----------

    Notes
    -----

    See Also
    --------

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

        self.group_sum_squares = self._sst()
        self.residual_sum_squares = self._sse()

        self.group_mean_squares = self._mst()
        self.residual_mean_squares = self._mse()

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
        total_mean = np.mean(self.design_matrix[:, 1])

        sst = 0

        for i, j in zip(group_n, group_means):
            sst += i[1] * (j[1] - total_mean) ** 2

        return sst

    def _mst(self):
        mst = self.group_sum_squares / self.group_degrees_of_freedom

        return mst

    def _mse(self):
        mse = self.residual_sum_squares / self.residual_degrees_of_freedom

        return mse

    def _fvalue(self):
        return self.group_mean_squares / self.residual_mean_squares

    def _pvalue(self):
        p = 1 - f.cdf(self.f_statistic,
                      self.group_degrees_of_freedom,
                      self.residual_degrees_of_freedom)

        return p

    def _group_statistics(self):
        group_means = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], np.mean)
        group_obs = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], len)
        group_variance = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], var)

        group_stats = {
            'Group Means': group_means,
            'Group Observations': group_obs,
            'Group Variance': group_variance
        }

        return group_stats

    def summary(self):
        anova_results = {
            'Test description': self.test_description,
            'F-statistic': self.f_statistic,
            'p-value': self.p_value,
            'Group DoF': self.group_degrees_of_freedom,
            'Residual DoF': self.residual_degrees_of_freedom,
            'Group Sum of Squares': self.group_sum_squares,
            'Group Mean Squares': self.group_mean_squares,
            'Residual Sum of Squares': self.residual_sum_squares,
            'Residual Mean Squares': self.residual_mean_squares,
            'Group statistics': self.group_statistics
        }

        return anova_results


class ManovaOneWay(object):

    def __init__(self, *args, group):

        self.design_matrix = build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.k = len(np.unique(self.group))

        self._group_stats = self._group_statistics()
        self._observation_stats = self._obs_statistics()

        self.hypothesis_matrix, self.error_matrix = self._hypothesis_error_matrix()
        self._intermediate_statistic_parameters = self._intermediate_test_statistic_parameters()

        self.degrees_of_freedom = self._degrees_of_freedom()
        self.numerator_dof = self.degrees_of_freedom['Numerator Degrees of Freedom']
        self.denominator_dof = self.degrees_of_freedom['Denominator Degrees of Freedom']
        self.pillai_statistic = self._pillai_statistic()
        self.wilks_lambda = self._wilks_statistic()
        self.roys_statistic = self._roys_statistic()
        self.hotelling_t2_statistic = self._hotelling_t2_statistic()
        self.test_description = 'One-Way MANOVA'

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

    def _hypothesis_error_matrix(self):

        groupmeans = self._group_stats['Group Means']
        xmeans = self._observation_stats['x means']

        n = self._group_stats['Group Observations']
        xn = len(xmeans)

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

    def _pillai_statistic(self):

        nn, s, m = self._intermediate_statistic_parameters['nn'], \
                   self._intermediate_statistic_parameters['s'], \
                   self._intermediate_statistic_parameters['m']

        pillai = np.sum(np.diag(np.dot(np.linalg.inv(self.hypothesis_matrix + self.error_matrix),
                                       self.hypothesis_matrix)))

        pillai_f = ((2. * nn + s + 1.) * pillai) / ((2. * m + s + 1.) * (s - pillai))

        pillai_stat = {'Pillai Statistic': pillai,
                       'Pillai F-value': pillai_f,
                       'Pillai p-value': self._f_p_value(pillai_f,
                                                         self.numerator_dof,
                                                         self.denominator_dof)}

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

        wilks_stat = {"Wilks Lambda": wilks_lambda,
                      "Wilks Lambda F-value": wilks_lambda_f,
                      "Wilks Lambda p-value": self._f_p_value(wilks_lambda_f,
                                                              self.numerator_dof,
                                                              self.denominator_dof)}

        return wilks_stat

    def _roys_statistic(self):
        eigs, n, vh = self._intermediate_statistic_parameters['eigs'], \
                      self._intermediate_statistic_parameters['n'], \
                      self._intermediate_statistic_parameters['vh']

        roy = np.max(eigs)
        roy_f = float(self.k * (n - 1)) / float(vh) * roy

        roy_stat = {"Roys Statistic": roy,
                    "Roys Statistic F-value": roy_f,
                    "Roys Statistic p-value": self._f_p_value(roy_f,
                                                              self.numerator_dof,
                                                              self.denominator_dof)}

        return roy_stat

    def _hotelling_t2_statistic(self):
        dot_inve_h, s, nn, m = self._intermediate_statistic_parameters['dot_inve_h'], \
                               self._intermediate_statistic_parameters['s'], \
                               self._intermediate_statistic_parameters['nn'], \
                               self._intermediate_statistic_parameters['m']

        t2 = np.sum(np.diag(dot_inve_h))
        t2_f = (2. * (s * nn + 1.) * np.sum(dot_inve_h)) / (s ** 2. * (2. * m + s + 1.))

        t2_stat = {
            "Hotellings T^2 Statistic": t2,
            "Hotellings T^2 F-value": t2_f,
            "Hotellings T^2 p-value": self._f_p_value(t2_f,
                                                      self.numerator_dof,
                                                      self.denominator_dof)
        }

        return t2_stat

    def _degrees_of_freedom(self):
        vh, ve, xn = self._intermediate_statistic_parameters['vh'], \
                     self._intermediate_statistic_parameters['ve'], \
                     len(self._observation_stats['x means'])

        num_df, denom_df = vh, ve

        dof = {'Numerator Degrees of Freedom': num_df,
               'Denominator Degrees of Freedom': denom_df}

        return dof

    def _f_p_value(self, f_val, df_num, df_denom):
        p = 1 - f.cdf(f_val, df_num, df_denom)

        return p

    @staticmethod
    def _dot_inve_h(h, e):
        return np.dot(np.linalg.inv(e), h)

    def summary(self):
        manova_result = {
            'Test Description': self.test_description,
            'degrees of freedom': self.degrees_of_freedom,
            'Pillai Statistic': self.pillai_statistic,
            "Wilks Lambda": self.wilks_lambda,
            "Roys Statistic": self.roys_statistic,
            "Hotellings T^2": self.hotelling_t2_statistic
        }

        return manova_result
