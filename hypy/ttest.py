# encoding=utf-8

import numpy as np
import patsy
from scipy.stats import t


def t_test(formula, data={}, mu=None, var_equal=False, paired=False, alternative='two-sided'):
    pass


class tTest(object):

    def __init__(self, formula, data={}, mu=None, var_equal=False, paired=False, alternative='two-sided'):

        if len(data.keys()) != 1:
            self.y1, self.y2 = patsy.dmatrices(formula, data, 1)
        else:
            self.y1 = patsy.dmatrix(formula, data, 1)
            self.y2 = None

        self.alternative = alternative
        self.mu = mu
        self.sample_statistics = {'y1_sample_statistics': self.sample_stats(self.y1)}

        if self.y2 is not None:
            self.sample_statistics['y2_sample_statistics'] = self.sample_stats(self.y2)

        if var_equal:
            self.method = "Student's t-test"
            self.var_equal = True
        else:
            self.method = "Welch's t-test"
            self.var_equal = var_equal

        self.parameter = self.degrees_of_freedom()
        self.t_statistic = self.test_statistic()
        self.p_value = self.pval()
        self.confidence_interval = self.conf_int()

    def pval(self):
        p = t.cdf(self.t_statistic, self.parameter)

        if self.alternative == 'two-sided':
            p *= 2.

        return p

    def test_statistic(self):
        n1, s1, ybar1 = self.sample_statistics['y1_sample_statistics']['obs'], \
                        self.sample_statistics['y1_sample_statistics']['variance'], \
                        self.sample_statistics['y1_sample_statistics']['mean']

        if self.y2 is not None:
            n2, s2, ybar2 = self.sample_statistics['y2_sample_statistics']['obs'], \
                            self.sample_statistics['y2_sample_statistics']['variance'], \
                            self.sample_statistics['y2_sample_statistics']['mean']

            if self.var_equal:
                sp = np.sqrt(((n1 - 1.) * s1 + (n2 - 1.) * s2) / (n1 + n2 - 2.))
                tval = float((ybar1 - ybar2) / (sp * np.sqrt(1. / n1 + 1. / n2)))
            else:
                tval = float((ybar1 - ybar2) / np.sqrt(s1 / n1 + s2 / n2))

        else:
            ybar2, n2, s2 = 0.0, 1.0, 0.0

            if self.mu is None:
                mu = 0.0
            else:
                mu = self.mu

            tval = float((ybar2 - mu) / np.sqrt(s2 / n2))

        return tval

    def conf_int(self):

        xn, xvar, xbar = self.sample_statistics['y1_sample_statistics']['obs'], \
                         self.sample_statistics['y1_sample_statistics']['variance'], \
                         self.sample_statistics['y1_sample_statistics']['mean']

        if self.y2 is not None:
            yn, yvar, ybar = self.sample_statistics['y2_sample_statistics']['obs'], \
                             self.sample_statistics['y2_sample_statistics']['variance'], \
                             self.sample_statistics['y2_sample_statistics']['mean']

            low_interval = (xbar - ybar) + t.ppf(0.025, self.parameter) * np.sqrt(xvar / xn + yvar / yn)
            high_interval = (xbar - ybar) - t.ppf(0.025, self.parameter) * np.sqrt(xvar / xn + yvar / yn)
        else:
            low_interval = xbar + 1.96 * np.sqrt((xbar * (1 - xbar)) / xn)
            high_interval = xbar - 1.96 * np.sqrt((xbar * (1 - xbar)) / xn)

        return float(low_interval), float(high_interval)

    def degrees_of_freedom(self):
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
        n1, s1 = self.sample_statistics['y1_sample_statistics']['obs'], self.sample_statistics['y1_sample_statistics'][
            'variance']

        v1 = n1 - 1

        if self.y2 is not None:
            n2, s2 = self.sample_statistics['y2_sample_statistics']['obs'], \
                     self.sample_statistics['y2_sample_statistics']['variance']

            v2 = n2 - 1

            if self.var_equal:
                v = n1 + n2 - 2
            else:
                v = np.power((s1 / n1 + s2 / n2), 2) / (np.power((s1 / n1), 2) / v1 + np.power((s2 / n2), 2) / v2)

        else:
            v = v1

        return float(v)

    @staticmethod
    def sample_stats(sample_vector):

        sample_stats = {
            'obs': len(sample_vector),
            'variance': np.var(sample_vector),
            'mean': np.mean(sample_vector)
        }

        return sample_stats
