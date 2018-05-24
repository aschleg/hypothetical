

import numpy as np
import numpy_indexed as npi
from collections import namedtuple
from scipy.stats import f


def anova_one_way(group, x, *args):

    return AnovaOneWay(group=group, x=x, *args)


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
    def __init__(self, group, x, *args):
        self.group = group
        self.x = x
        self.design_matrix = _build_des_mat(group, x, *args)
        self.group_statistics = self._group_statistics()
        self.group_names = np.unique(self.group)
        self.k = len(self.group_names)
        self.group_degrees_of_freedom = self.k - 1
        self.residual_degrees_of_freedom = len(x) - self.k
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

    def __init__(self, group, x, *args):
        self.group = group
        self.x = x
        self.design_matrix = _build_des_mat(group, x, args)
        #self.group_names = np.unique(self.group)
        self.k = len(np.unique(self.group))
        self.hypothesis_matrix, self.error_matrix = self._hypothesis_error_matrix()

        self._group_stats = self._group_statistics()
        self._observation_stats = self._obs_statistics()

    def _group_statistics(self):
        group_means = npi.group_by(self.design_matrix[:, 0]).mean(self.design_matrix)[1][:, 1:]
        group_observations = npi.group_by(self.design_matrix[:, 0], self.design_matrix, len)
        groups = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1:])[1]

        group_stats = {
            'Group Means': group_means,
            'Group Observations': group_observations,
            'Groups': groups
        }

        return group_stats

    def _obs_statistics(self):
        x_means = self.design_matrix[:, 1:].mean(axis=0)
        x_observations = len(x_means)

        obs_stats = {
            'x means': x_means,
            'x observations': x_observations
        }

        return obs_stats

    def _hypothesis_error_matrix(self):

        groupmeans = self._group_stats['Group Means']
        xmeans = self._observation_stats['x means']

        xn = self._observation_stats['x observations']
        groupn = self._group_stats['Group Observations']

        n = [i for _, i in groupn]

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
        p = len(self.error_matrix)
        n = len(self.x)

        vh = self.k - 1.
        ve = n - self.k

        s = np.minimum(vh, p)
        m = 0.5 * (np.absolute(vh - p) - 1)
        nn = 0.5 * (ve - p - 1)

        

    def _pillai_statistic(self):
        pillai = np.sum(np.diag(np.dot(np.linalg.inv(self.hypothesis_matrix + self.error_matrix),
                                       self.hypothesis_matrix)))

        pillai_f = ((2. * nn + s + 1.) * pillai) / ((2. * m + s + 1.) * (s - pillai))

        return pillai, pillai_f

    def _wilks_statistic(self):
        dot_inve_h = self._dot_inve_h(self.hypothesis_matrix, self.error_matrix)
        eigs = np.linalg.eigvals(dot_inve_h)

    def _roys_statistic(self):
        pass

    def _hotelling_t2_statistic(self):
        pass

    @staticmethod
    def _dot_inve_h(h, e):
        return np.dot(np.linalg.inv(e), h)


def manova_oneway(group, x, *args):
    r"""
    Performs Multiple Analysis of Variance (MANOVA) of one grouping variable and n dependent variables

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
        Namedtuple with the following entries representing a MANOVA table:
        Group Df: Group Vector Degrees of Freedom
        residual Df: Residuals Degrees of Freedom
        Num Df: Numerator Degrees of Freedom
        Den Df: Denominator Degrees of Freedom
        Pillai Statistic: Pillai Test Statistic
        Wilk's Lambda: Wilk's Lambda
        Lawley-Hotelling T^2: T^2 statistic, also known as Lawley-Hotelling statistic
        Roy's Test: Reported value from Roy's Test
        Pillai F-Value: Approximated F-Value of Pillai statistic
        Wilk's Lambda F-Value: Approximated F-Value of Wilk's Lambda
        Lawley-Hotelling T^2 F-Value: Approximated F-Value of T^2
        Roy's Test F-Value: Approximated F-Value of Roy's Test statistic
        Pillai p-value: p-value of approximated Pillai F-Value with Num Df and Den Df
        Wilk's Lambda p-value: p-value of approximated Wilk's Lambda F-Value with Num Df and Den Df
        Lawley-Hotelling T^2 p-value: p-value of approximated Lawley-Hotelling F-Value with Num Df and Den Df
        Roy's Test p-value: p-value of approximated Roy's Test F-Value with Num Df and Den Df

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

    References
    ----------
    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    """
    if args is not ():
        c = args[0]
        for i in np.arange(1, len(args)):
            c = np.column_stack((c, args[i]))
        x = np.column_stack((x, c))

    x = _create_array(x)[0]
    grouparr, groupname = _create_array(group)

    groupnames = np.unique(grouparr)
    kn = len(groupnames)

    data = np.column_stack((grouparr, x))

    xmeans = data[:, 1:].mean(axis=0)
    xn = len(xmeans)

    groupmeans = npi.group_by(data[:, 0]).mean(data)[1][:, 1:]
    groupn = npi.group_by(data[:, 0], data, len)

    groups = npi.group_by(data[:, 0], data[:, 1:])[1]

    n = [i for _, i in groupn]

    h, e = np.zeros((xn, xn)), np.zeros((xn, xn))

    for i in np.arange(xn):
        for j in np.arange(i + 1):

            h[i, j] = n[i] * np.sum((groupmeans[:, i] - xmeans[i]) * (groupmeans[:, j] - xmeans[j]))
            h[j, i] = n[i] * np.sum((groupmeans[:, j] - xmeans[j]) * (groupmeans[:, i] - xmeans[i]))

            b = []

            for k in groups:
                a = np.sum((k[:, i] - np.mean(k[:, i])) * (k[:, j] - np.mean(k[:, j])))
                b.append(a)

            e[i, j], e[j, i] = np.sum(b), np.sum(b)

    vh, ve, pillai, pillai_f, wilks_lambda, wilks_lambda_f, t2, t2_f, roy, roy_f = _manova_statistics(h, e, kn, len(x))

    num_df, denom_df = vh * xn, ve * xn

    pillai_pval, wilks_pval, t2_pval, roy_pval = _f_p_value(pillai_f, num_df, denom_df), \
                                                 _f_p_value(wilks_lambda_f, num_df, denom_df), \
                                                 _f_p_value(t2_f, num_df, denom_df), \
                                                 _f_p_value(roy_f, num_df, denom_df)

    ManovaResult = namedtuple('ManovaResult', ['groupdf', 'residualdf', 'numdf', 'denomdf',
                                               'pillai', 'wilks', 't2', 'roy',
                                               'pillai_f', 'wilks_f', 't2_f', 'roy_f',
                                               'pillai_p', 'wilks_p', 't2_p', 'roy_p'])

    maov = ManovaResult(groupdf=vh, residualdf=ve, numdf=num_df, denomdf=denom_df,
                        pillai=pillai, wilks=wilks_lambda, t2=t2, roy=roy,
                        pillai_f=pillai_f, wilks_f=wilks_lambda_f, t2_f=t2_f, roy_f=roy_f,
                        pillai_p=pillai_pval, wilks_p=wilks_pval, t2_p=t2_pval, roy_p=roy_pval)

    return maov


def _manova_statistics(h, e, k, n):
    r"""
    Helper function that computes several MANOVA test statistics. Not meant to be called outside of
    the :code:`manova()` function.

    Parameters
    ----------
    h
        The hypothesis matrix
    e
        The error matrix
    k
        Number of groups
    n
        Total sample size

    Returns
    -------
    list

    Notes
    -----
    Multiple tests of significance can be employed when performing MANOVA. The
    most well known and widely used MANOVA test statistics are Wilk's :math:`\Lambda`,
    Pillai, Lawley-Hotelling, and Roy's test. Unlike ANOVA, in which only one
    dependent variable is examined, several tests are often utilized in MANOVA
    due to its multidimensional nature. Each MANOVA test statistic is different
    and can lead to different conclusions depending on how the data and mean vectors lie.

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

    The Lawley-Hotelling statistic, also known as Hotelling's generalized :math:`T^2`-statistic,
    is denoted :math:`U^{(s)}` and is defined as:

    .. math::

        U^{(s)} = tr(E^{-1}H) = \sum^s_{i=1} \lambda_i

    The approximate F-statistic in the Lawley-Hotelling setting is defined as:

    .. math::

        F = \frac{2 (sN + 1)U^{(s)}}{s^2(2m + s + 1)}

    Where :math:`s`, :math:`m`, and :math`N` are defined as above.

    Roy's test statistic is the largest eigenvalue of the matrix :math:`E^{-1}H`

    The F-statistic in Roy's setting can be computed as:

    .. math::

        \frac{k(n - 1)}{k - 1} \lambda_1

    References
    ----------
    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    See Also
    --------
    manova
        Function for performing MANOVA (Multiple Analysis of Variance procedure)

    """
    dot_inve_h = _dot_inve_h(h, e)
    eigs = np.linalg.eigvals(dot_inve_h)

    p = len(e)

    vh = k - 1.
    ve = n - k

    s = np.minimum(vh, p)
    m = 0.5 * (np.absolute(vh - p) - 1)
    nn = 0.5 * (ve - p - 1)

    t = np.sqrt((p ** 2. * vh ** 2. - 4.) / (p ** 2. + vh ** 2. - 5.))
    df1 = p * vh
    df2 = (ve + vh - .5 * (p + vh + 1.)) * t - .5 * (p * vh - 2.)

    pillai = np.sum(np.diag(np.dot(np.linalg.inv(h + e), h)))
    pillai_f = ((2. * nn + s + 1.) * pillai) / ((2. * m + s + 1.) * (s - pillai))

    wilks_lambda = np.prod(1. / (1. + eigs))
    wilks_lambda_f = ((1. - wilks_lambda ** (1. / t)) / wilks_lambda ** (1. / t)) * (df2 / df1)

    t2 = np.sum(np.diag(dot_inve_h))
    t2_f = (2. * (s * nn + 1.) * np.sum(dot_inve_h)) / (s ** 2. * (2. * m + s + 1.))

    roy = np.max(eigs)
    roy_f = float(k * (n - 1)) / float(vh) * roy

    return [vh, ve, pillai, pillai_f, wilks_lambda, wilks_lambda_f, t2, t2_f, roy, roy_f]


def _build_des_mat(group, x, *args):

    if args is not ():
        c = args[0]
        for i in np.arange(1, len(args)):
            c = np.column_stack((c, args[i]))
        mat = np.column_stack((x, c))

    else:
        mat = x.copy()

    if mat.ndim > 1:
        mat = np.sum(mat, axis=1)

    data = np.column_stack([group, mat])

    return data
