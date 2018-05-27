# encoding=utf-8

import numpy as np
import numpy_indexed as npi
from scipy.stats import t


def t_test(y1, y2=None, group=None, mu=None, var_equal=False, paired=False, alternative='two-sided'):

    return tTest(y1=y1, y2=y2, group=group, mu=mu, var_equal=var_equal, paired=paired, alternative=alternative)


class tTest(object):

    def __init__(self, y1, y2=None, group=None, mu=None, var_equal=False, paired=False, alternative='two-sided'):
        self.group = group
        self.paired = paired

        if alternative not in ('two-sided', 'greater', 'less'):
            raise ValueError("alternative must be one of 'two-sided', 'greater', or 'lesser'")

        self.alternative = alternative

        if self.paired and y2 is None:
            raise ValueError('second sample is missing for paired test')

        self.mu = mu

        if var_equal:
            self.method = "Student's t-test"
            self.var_equal = True
        else:
            self.method = "Welch's t-test"
            self.var_equal = var_equal

        if self.paired:
            self.test_description = 'Paired t-test'
            self.y1 = self._paired(y1, y2)
            self._y1_summary_stat_name = 'Sample Difference'
            self.y2 = None
            self.sample_statistics = {self._y1_summary_stat_name: self._sample_stats(self.y1)}
        elif y2 is None and group is None:
            self.test_description = 'One-Sample t-test'
            self._y1_summary_stat_name = 'Sample 1'
            self.y1, self.y2 = y1, None
            self.sample_statistics = {self._y1_summary_stat_name: self._sample_stats(self.y1)}
        else:
            self.test_description = 'Two-Sample' + ' ' + self.method
            self._y1_summary_stat_name = 'Sample 1'

            if group is None:
                self.y1, self.y2 = y1, y2
            else:
                self.y1, self.y2 = self._split_groups(y1)

            self.sample_statistics = {self._y1_summary_stat_name: self._sample_stats(self.y1)}

            self.sample_statistics['Sample 2'] = self._sample_stats(self.y2)

        self.parameter = self._degrees_of_freedom()
        self.t_statistic = self._test_statistic()
        self.p_value = self._pvalue()
        self.confidence_interval = self._conf_int()

    def summary(self):
        test_results = {
            't-statistic': self.t_statistic,
            'p-value': self.p_value,
            'confidence interval': self.confidence_interval,
            'degrees of freedom': self.parameter,
            'alternative': self.alternative,
            'test description': self.test_description,
            self._y1_summary_stat_name + ' Mean': self.sample_statistics[self._y1_summary_stat_name]['mean']
        }

        if self.y2 is not None:
            test_results['Sample 2 Mean'] = self.sample_statistics['Sample 2']['mean']

        if self.mu is not None:
            test_results['mu'] = self.mu

        return test_results

    def _degrees_of_freedom(self):
        r"""
        Computes the degrees of freedom of one or two samples.

        Parameters
        ----------
        y1
            First sample to test
        y2
            Second sample. Optional.
        var_equal
            Optional, default False. If False, Welch's t-test for unequal variance and
            sample sizes is used. If True, equal variance between samples is assumed
            and Student's t-test is used.

        Returns
        -------
        float
            the degrees of freedom

        Notes
        -----
        When Welch's t test is used, the Welch-Satterthwaite equation for approximating the degrees
        of freedom should be used and is defined as:

        .. math::

            \large v \approx \frac{\left(\frac{s_{1}^2}{N_1} +
            \frac{s_{2}^2}{N_2}\right)^2}{\frac{\left(\frac{s_1^2}{N_1^{2}}\right)^2}{v_1} +
            \frac{\left(\frac{s_2^2}{N_2^{2}}\right)^2}{v_2}}

        If the two samples are assumed to have equal variance, the degrees of freedoms become simply:

        .. math::

            v = n_1 + n_2 - 2

        In the case of one sample, the degrees of freedom are:

        .. math::

            v = n - 1

        References
        ----------
        Rencher, A. C., & Christensen, W. F. (2012). Methods of multivariate analysis (3rd Edition).

        Welch's t-test. (2017, June 16). In Wikipedia, The Free Encyclopedia.
            From https://en.wikipedia.org/w/index.php?title=Welch%27s_t-test&oldid=785961228

        """
        n1, s1 = self.sample_statistics[self._y1_summary_stat_name]['obs'], \
                 self.sample_statistics[self._y1_summary_stat_name]['variance']

        v1 = n1 - 1

        if self.y2 is not None:
            n2, s2 = self.sample_statistics['Sample 2']['obs'], \
                     self.sample_statistics['Sample 2']['variance']

            v2 = n2 - 1

            if self.var_equal:
                v = n1 + n2 - 2
            else:
                v = np.power((s1 / n1 + s2 / n2), 2) / (np.power((s1 / n1), 2) / v1 + np.power((s2 / n2), 2) / v2)

        else:
            v = v1

        return float(v)

    def _test_statistic(self):
        n1, s1, ybar1 = self.sample_statistics[self._y1_summary_stat_name]['obs'], \
                        self.sample_statistics[self._y1_summary_stat_name]['variance'], \
                        self.sample_statistics[self._y1_summary_stat_name]['mean']

        if self.y2 is not None:
            n2, s2, ybar2 = self.sample_statistics['Sample 2']['obs'], \
                            self.sample_statistics['Sample 2']['variance'], \
                            self.sample_statistics['Sample 2']['mean']

            if self.var_equal:
                sp = np.sqrt(((n1 - 1.) * s1 + (n2 - 1.) * s2) / (n1 + n2 - 2.))
                tval = float((ybar1 - ybar2) / (sp * np.sqrt(1. / n1 + 1. / n2)))
            else:
                tval = float((ybar1 - ybar2) / np.sqrt(s1 / n1 + s2 / n2))

        else:

            if self.mu is None:
                mu = 0.0
            else:
                mu = self.mu

            tval = float((ybar1 - mu) / np.sqrt(s1 / n1))

        return tval

    def _pvalue(self):
        p = t.cdf(self.t_statistic, self.parameter)

        if self.alternative == 'two-sided':
            p *= 2.
        elif self.alternative == 'greater':
            p = 1 - p

        if 1.0 < p < 2.0:
            p = 2 - p

        if p == 2.0:
            p = np.finfo(float).eps

        return p

    def _conf_int(self):
        xn, xvar, xbar = self.sample_statistics[self._y1_summary_stat_name]['obs'], \
                         self.sample_statistics[self._y1_summary_stat_name]['variance'], \
                         self.sample_statistics[self._y1_summary_stat_name]['mean']

        if self.y2 is not None:
            yn, yvar, ybar = self.sample_statistics['Sample 2']['obs'], \
                             self.sample_statistics['Sample 2']['variance'], \
                             self.sample_statistics['Sample 2']['mean']

            low_interval = (xbar - ybar) + t.ppf(0.025, self.parameter) * np.sqrt(xvar / xn + yvar / yn)
            high_interval = (xbar - ybar) - t.ppf(0.025, self.parameter) * np.sqrt(xvar / xn + yvar / yn)

        else:
            low_interval = xbar + 1.96 * np.sqrt(xvar / xn)
            high_interval = xbar - 1.96 * np.sqrt(xvar / xn)

        if self.alternative == 'greater':
            intervals = np.inf, float(high_interval)
        elif self.alternative == 'less':
            intervals = -np.inf, float(low_interval)
        else:
            intervals = float(low_interval), float(high_interval)

        return intervals

    def _split_groups(self, x):
        if len(np.unique(self.group)) > 2:
            raise ValueError('there cannot be more than two groups')

        obs_matrix = npi.group_by(self.group, x)

        y1 = obs_matrix[1][0]
        self.sample_statistics = {'y1_sample_statistics': self._sample_stats(y1)}

        if len(obs_matrix[0]) == 2:
            y2 = obs_matrix[1][1]
            self.sample_statistics['y2_sample_statistics'] = self._sample_stats(y2)
        else:
            y2 = None

        return y1, y2

    @staticmethod
    def _paired(y1, y2):

        if len(y1) != len(y2):
            raise ValueError('paired samples must have the same number of observations')

        x = np.array(y1) - np.array(y2)

        return x

    @staticmethod
    def _sample_stats(sample_vector):
        sample_stats = {
            'obs': int(len(sample_vector)),
            'variance': float(np.var(sample_vector)),
            'mean': float(np.mean(sample_vector))
        }

        return sample_stats
