# encoding=utf8

"""
Functions for performing post-hoc analysis.

Post-Hoc Analysis
-----------------

..autosummary::
    :toctree: generated/

    GamesHowell
    TukeysTest

References
----------

Ruxton, G.D., and Beauchamp, G. (2008) 'Time for some a priori thinking about post hoc testing',
    Behavioral Ecology, 19(3), pp. 690-693. doi: 10.1093/beheco/arn020.
    In-text citations: (Ruxton and Beauchamp, 2008)

Post-hoc (no date) Available at: http://www.unt.edu/rss/class/Jon/ISSS_SC/Module009/isss_m91_onewayanova/node7.html

"""

from hypothetical._lib import _build_des_mat
import numpy as np
import pandas as pd
import numpy_indexed as npi
from hypothetical.descriptive import var, std_dev
from statsmodels.stats.libqsturng import qsturng, psturng
from itertools import combinations


class GamesHowell(object):
    r"""
    Computes the Games-Howell posthoc analysis test for multiple group comparisons.

    Parameters
    ----------
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector.
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector.
    alpha : float, default 0.05
        Alpha level for computing upper and lower confidence intervals

    Attributes
    ----------
    design_matrix : array-like
        Numpy ndarray representing the data matrix for the analysis.
    group : array-like
        Group vector passed or coerced from sample observation vectors.
    alpha : float, default 0.05
        Alpha level for computing upper and lower confidence intervals
    test_result : array-like
        pandas DataFrame of test results containing each group comparison's mean difference,
        confidence interval, t-value, p-value and standard error.

    Notes
    -----
    The Games-Howell post-hoc test is another nonparametric approach to compare combinations of groups
    or treatments. Although rather similar to Tukey's test in its formulation, the Games-Howell test
    does not assume equal variances and sample sizes. The test was designed based on Welch's degrees of
    freedom correction and uses Tukey's studentized range distribution, denoted :math:`q`. The Games-Howell
    test is performed on the ranked variables similar to other nonparametric tests. Since the Games-Howell
    test does not rely on equal variances and sample sizes, it is often recommended over other approaches
    such as Tukey's test.

    The Games-Howell test is defined as:

    .. math::

        \large \bar{x}_i - \bar{x}_j > q_{\sigma, k, df}

    Where :math:`\sigma` is equal to standard error:

    .. math::

        \sigma = \sqrt{{\frac{1}{2} \left(\frac{s^2_i}{n_i} + \frac{s^2_j}{n_j}\right)}}

    Degrees of freedom is calculated using Welch's correction:

    .. math::

        \large \frac{\left(\frac{s^2_i}{n_i} +
        \frac{s^2_j}{n_j}\right)^2}{\frac{\left(\frac{s_i^2}{n_i}\right)^2}{n_i - 1} +
        \frac{\left(\frac{s_j^2}{n_j}\right)^2}{n_j - 1}}

    Thus, confidence intervals can be formed with:

    .. math::

        \bar{x}_i - \bar{x}_j \pm t \sqrt{{\frac{1}{2} \left(\frac{s_i^2}{n_i} + \frac{s_j^2}{n_j}\right)}}

    p-values are calculated using Tukey's studentized range:

    .. math::

        \large q_{t * \sqrt{2}, k, df}

    The Games-Howell test and Tukey's test will often report similar results with data that is assumed to have
    equal variance and equal sample sizes.

    References
    ----------
    Ruxton, G.D., and Beauchamp, G. (2008) 'Time for some a priori thinking about post hoc testing',
        Behavioral Ecology, 19(3), pp. 690-693. doi: 10.1093/beheco/arn020.
        In-text citations: (Ruxton and Beauchamp, 2008)

    Post-hoc (no date) Available at: http://www.unt.edu/rss/class/Jon/ISSS_SC/Module009/isss_m91_onewayanova/node7.html

    """
    def __init__(self, *args, group, alpha=0.05):
        self.design_matrix = _build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.alpha = alpha
        self.test_result = self._games_howell_test()

    def _group_sample_statistics(self):
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

        groups = len(np.unique(self.design_matrix[:, 0]))

        group_stats = {
            'Group Means': group_means,
            'Group Observations': group_obs,
            'Group Variance': group_variance,
            'Number of Groups': groups
        }

        return group_stats

    def _games_howell_test(self):
        combs = list(combinations(np.unique(self.group), 2))
        sample_stats = self._group_sample_statistics()

        means_d = dict(sample_stats['Group Means'])
        obs_d = dict(sample_stats['Group Observations'])
        var_d = dict(sample_stats['Group Variance'])

        group_comps = []
        mean_differences = []
        degrees_freedom = []
        t_values = []
        p_values = []
        std_err = []
        up_conf = []
        low_conf = []

        for comb in combs:

            diff = means_d[comb[1]] - means_d[comb[0]]

            t_val = np.absolute(diff) / np.sqrt((var_d[comb[0]] / obs_d[comb[0]]) + (var_d[comb[1]] / obs_d[comb[1]]))

            df_num = (var_d[comb[0]] / obs_d[comb[0]] + var_d[comb[1]] / obs_d[comb[1]]) ** 2
            df_denom = ((var_d[comb[0]] / obs_d[comb[0]]) ** 2 / (obs_d[comb[0]] - 1) +
                        (var_d[comb[1]] / obs_d[comb[1]]) ** 2 / (obs_d[comb[1]] - 1))

            df = df_num / df_denom

            p_val = psturng(t_val * np.sqrt(2), sample_stats['Number of Groups'], df)

            se = np.sqrt(0.5 * (var_d[comb[0]] / obs_d[comb[0]] + var_d[comb[1]] / obs_d[comb[1]]))

            upper_conf = diff + qsturng(1 - self.alpha, sample_stats['Number of Groups'], df)
            lower_conf = diff - qsturng(1 - self.alpha, sample_stats['Number of Groups'], df)

            mean_differences.append(diff)
            degrees_freedom.append(df)
            t_values.append(t_val)
            p_values.append(p_val)
            std_err.append(se)
            up_conf.append(upper_conf)
            low_conf.append(lower_conf)
            group_comps.append(str(comb[0]) + ' : ' + str(comb[1]))

        result_df = pd.DataFrame({'groups': group_comps,
                                  'mean_difference': mean_differences,
                                  'std_error': std_err,
                                  't_value': t_values,
                                  'p_value': p_values,
                                  'upper_limit': up_conf,
                                  'lower limit': low_conf})

        return result_df


class TukeysTest(object):
    r"""
    Performs Tukey's Honestly Significant Difference posthoc analysis test.

    Parameters
    ----------
    group: array-like, optional
        One-dimensional array (Numpy ndarray, Pandas Series, list) that defines the group
        membership of the dependent variable(s). Must be the same length as the observation vector.
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector.
    alpha : float, default 0.05
        Alpha level for computing upper and lower confidence intervals

    Attributes
    ----------
    design_matrix : array-like
        Numpy ndarray representing the data matrix for the analysis.
    group : array-like
        Group vector passed or coerced from sample observation vectors.
    alpha : float, default 0.05
        Alpha level for computing upper and lower confidence intervals
    test_description : str, default 'Tukey multiple comparison of means'
        String denoting the type of procedure performed.
    n : int
        Number of total observations.
    k : int
        Number of groups
    dof : int
        Degrees of freedom
    tukey_q_value : float
        The computed q-value used in determining the HSD
    mse : float
        The mean square error
    hsd : float
        Tukey's Honestly Significant Difference value
    group_comparison : array-like
        pandas DataFrame of group comparison results calculated by Tukey's HSD test.
    test_summary : dict
        Dictionary of test results.

    Notes
    -----
    After a multivariate test, it is often desired to know more about the specific groups to find out if they
    are significantly different or similar. This step after analysis is referred to as 'post-hoc analysis' and
    is a major step in hypothesis testing.

    One common and popular method of post-hoc analysis is Tukey's Test. The test is known by several different names.
    Tukey's test compares the means of all treatments to the mean of every other treatment and is considered the best
    available method in cases when confidence intervals are desired or if sample sizes are unequal.

    The test statistic used in Tukey's test is denoted :math:`q` and is essentially a modified t-statistic that
    corrects for multiple comparisons.

    :math:`q` can be found similarly to the t-statistic:

    .. math::

        q_{\alpha,k,N-k}

    The studentized range distribution of :math:`q` is defined as:

    .. math::

        q_s = \frac{Y_{max} - Y_{min}}{SE}

    Where :math:`Y_{max}` and :math:`Y_{min}` are the larger and smaller means of the two groups being compared.
    :math:`SE` is defined as the standard error of the entire design.

    The Honestly Significant Difference is defined as the q-value multiplied by the square root of the MSE
    divided by the sample size.

    .. math::

        HSD = q_{\alpha,k,N-k} \sqrt{\frac{MSE}{n}}

    If the absolute value of the difference of the two groups' means is greater than or equal to the HSD,
    the difference is significant.

    .. math::

        \left|Y_1 - Y_2\right| \geq HSD

    Intervals for Tukey's Test can also be estimated. Since the test uses the studentized range, estimation is
    similar to the t-test setting. Intervals with :math:`1 - \alpha` confidence can be found using the
    Tukey-Kramer method. The Tukey-Kramer method allows for unequal sample sizes between the treatments and is,
    therefore, more often applicable. The Tukey-Kramer method is defined as:

    .. math::

        y_i - y_j \pm q_{\alpha,k,N-k} \sqrt{\left(\frac{MSE}{2}\right) \left(\frac{1}{n_i} + \frac{1}{n_j}\right)}

    Examples
    --------

    References
    ----------

    """
    def __init__(self, *args, group=None, alpha=0.95):
        self.alpha = alpha
        self.test_description = 'Tukey multiple comparisons of means'

        self.design_matrix = _build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.n = self.design_matrix.shape[0]
        self.k = len(np.unique(self.design_matrix[:, 0]))
        self.dof = self.n - self.k
        self.tukey_q_value = self._qvalue()
        self.mse = self._mse()
        self.hsd = self._hsd()
        self.group_comparison = self._group_comparison()
        self.test_summary = {
            'test description': self.test_description,
            'HSD': self.hsd,
            'MSE': self.mse,
            'Studentized Range q-value': self.tukey_q_value,
            'degrees of freedom': self.dof,
            'group comparisons': self.group_comparison.to_dict(),
            'alpha': self.alpha
        }

    def _mse(self):
        r"""
        Calculates the Mean Square Error for use in computing Tukey's HSD.

        Returns
        -------
        mse : float
            The Mean Square Error of the design.

        Notes
        -----
        The MSE is found by diving the standard error of the design by the total number of observations minus
        the number of groups.

        .. math::

            MSE = \frac{SE}{(N - k)}

        """
        group_variance = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], var)
        group_n = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], len)

        sse = 0

        for i, j in zip(group_n, group_variance):
            sse += (i[1] - 1) * j[1]

        mse = sse / (self.n - self.k)

        return mse

    def _qvalue(self):
        r"""
        Computes the q-value.

        Returns
        -------
        q : float
            The q-value

        Notes
        -----
        :math:`q` can be found similarly to the t-statistic:

        .. math::

            q_{\alpha,k,N-k}

        The studentized range distribution of :math:`q` is defined as:

        .. math::

            q_s = \frac{Y_{max} - Y_{min}}{SE}

        Where :math:`Y_{max}` and :math:`Y_{min}` are the larger and smaller means of the two groups being compared.
        :math:`SE` is defined as the standard error of the entire design.

        """
        q = qsturng(self.alpha, self.k, self.n - self.k)

        return q

    def _pvalue(self):
        pass

    def _hsd(self):
        r"""
        Computes Tukey's Honestly Significant Difference

        Returns
        -------
        hsd : float
            Tukey's Honestly Significant Difference

        Notes
        -----
        The Honestly Significant Difference is defined as the q-value multiplied by the square root of the MSE
        divided by the sample size.

        .. math::

            HSD = q_{\alpha,k,N-k} \sqrt{\frac{MSE}{n}}

        If the absolute value of the difference of the two groups' means is greater than or equal to the HSD,
        the difference is significant.

        """
        hsd = self.tukey_q_value * np.sqrt(self.mse / (self.n / self.k))

        return hsd

    def _group_comparison(self):
        r"""
        Constructs a pandas DataFrame containing the test results and group comparisons as found by Tukey's HSD test.

        Returns
        -------
        groups : array-like
            pandas DataFrame of group comparison results.

        """
        group_means = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], np.mean)

        group_means = [i for _, i in group_means]

        group_mean_differences = np.array(list(combinations(group_means, 2)))[:, 0] - \
                                 np.array(list(combinations(group_means, 2)))[:, 1]

        group_sd = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], std_dev)
        group_sd = [i for _, i in group_sd]

        group_names = np.unique(self.design_matrix[:, 0])

        groups = pd.DataFrame(np.array(list(combinations(group_names, 2))))

        groups['groups'] = groups[0] + ' - ' + groups[1]
        groups['group means'] = group_means
        groups['mean difference'] = group_mean_differences

        groups['std_dev'] = group_sd

        groups['significant difference'] = np.where(np.abs(groups['mean difference']) >= self.hsd, True, False)

        groups['upper interval'] = groups['mean difference'] + self.tukey_q_value * np.sqrt(
            self.mse / 2. * (2. / (self.n / self.k)))

        groups['lower interval'] = groups['mean difference'] - self.tukey_q_value * np.sqrt(
            self.mse / 2. * (2. / (self.n / self.k)))

        q_values = groups['mean difference'] / group_sd

        groups['p_adjusted'] = psturng(np.absolute(q_values), self.n / self.k, self.dof)

        del groups[0]
        del groups[1]

        return groups
