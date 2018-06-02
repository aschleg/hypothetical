import numpy as np
import numpy_indexed as npi
from scipy.stats import rankdata, norm, chi2, t

from hypothetical._lib import build_des_mat
from hypothetical.summary import var


def mann_whitney(y1, y2=None, group=None, continuity=True):
    r"""
    Performs the nonparametric Mann-Whitney U test of two independent sample groups.

    Parameters
    ----------
    y1
        One-dimensional array-like (Pandas Series or DataFrame, Numpy array, list, or dictionary)
        designating first sample
    y2
        One-dimensional array-like (Pandas Series or DataFrame, Numpy array, list, or dictionary)
        designating second sample to compare to first
    continuity
        Boolean, optional. If True, apply the continuity correction of :math:`\frac{1}{2}` to the
        mean rank.

    Returns
    -------
    namedtuple
        Namedtuple of following entries that contain resulting Mann-Whitney test statistics.
        Mann-Whitney U Test Statistic: The U Statistic of the Mann-Whitney test
        Mean Rank: The mean rank of U statistic
        Sigma: the standard deviation of U
        z-value: The standardized value of U
        p-value: p-value of U statistic compared to critical value

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

    References
    ----------
    Mann–Whitney U test. (2017, June 20). In Wikipedia, The Free Encyclopedia.
        From https://en.wikipedia.org/w/index.php?title=Mann%E2%80%93Whitney_U_test&oldid=786593885

    """
    if y2 is None and group is None:
        res = wilcox_test(y1=y1)
    else:
        res = MannWhitney(y1=y1, y2=y2, group=group, continuity=continuity)

    return res


def wilcox_test(y1, y2=None, paired=False, median=0, continuity=True):
    if y2 is not None and paired is False:
        res = MannWhitney(y1=y1, y2=y2, continuity=continuity)
    else:
        res = WilcoxonTest(y1=y1, y2=y2, paired=paired, median=median, continuity=continuity)

    return res


def kruskal_wallis(*args, group=None, alpha=0.05):
    return KruskalWallis(*args, group=group, alpha=alpha)


class MannWhitney(object):

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
        self._ranks = self._rank()
        self.U = self._u()
        self.meanrank = self._mu()
        self.sigma = self._sigma()
        self.z_value = self._zvalue()
        self.p_value = self._pvalue()

    def _rank(self):
        ranks = np.concatenate((self.y1, self.y2))

        ranks = rankdata(ranks, 'average')

        ranks = ranks[:self.n1]

        return ranks

    def _u(self):
        u1 = self.n1 * self.n2 + (self.n1 * (self.n1 + 1)) / 2. - np.sum(self._ranks)
        u2 = self.n1 * self.n2 - u1

        u = np.minimum(u1, u2)

        return u

    def _mu(self):

        mu = (self.n1 * self.n2) / 2. + (0.5 * self.continuity)

        return mu

    def _sigma(self):
        rankcounts = np.unique(self._ranks, return_counts=True)[1]

        sigma = np.sqrt(((self.n1 * self.n2) * (self.n + 1)) / 12. * (
                    1 - np.sum(rankcounts ** 3 - rankcounts) / float(self.n ** 3 - self.n)))

        return sigma

    def _zvalue(self):
        z = (np.absolute(self.U - self.meanrank)) / self.sigma

        return z

    def _pvalue(self):
        p = 1 - norm.cdf(self.z_value)

        return p * 2

    def summary(self):
        mw_results = {
            'continuity': self.continuity,
            'U': self.U,
            'mu meanrank': self.meanrank,
            'sigma': self.sigma,
            'z-value': self.z_value,
            'p-value': self.p_value,
            'test description': 'Mann-Whitney U test'
        }

        return mw_results


class WilcoxonTest(object):

    def __init__(self, y1, y2=None, paired=False, median=0, continuity=True, alpha=0.05, alternative='two-sided'):
        self.paired = paired
        self.median = median
        self.continuity = continuity
        self.test_description = 'Wilcoxon signed rank test'

        if paired:
            if y2 is None:
                raise ValueError('sample 2 is missing for paired test')
            if len(y1) != len(y2):
                raise ValueError('samples must have same length for paired test')

            self.y1 = np.array(y1) - np.array(y2)

        else:
            self.y1 = y1

        self.n = len(self.y1)

        self.V = self._test()

        self.z = self._zvalue()
        self.p = self._pvalue()

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

    def summary(self):
        test_results = {
            'V': self.V,
            'z-value': self.z,
            'p-value': self.p,
            'test description': self.test_description
        }

        return test_results

    def _test(self):
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
        sigma_w = np.sqrt((self.n * (self.n + 1) * (2 * self.n + 1)) / 6.)

        z = self.V / sigma_w

        return z

    def _pvalue(self):
        p = (1 - norm.cdf(np.abs(self.z))) * 2

        if p == 0:
            p = np.finfo(float).eps

        return p


class KruskalWallis(object):

    def __init__(self, *args, group=None, alpha=0.05):

        if group is not None and len(args) > 1:
            raise ValueError('Only one sample vector should be passed when including a group vector')

        self.design_matrix = build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]

        self.ranked_matrix = self._rank()
        self.group_rank_sums = self._group_rank_sums()
        self.alpha = alpha
        self.n = self.design_matrix.shape[0]
        self.k = len(np.unique(self.design_matrix[:, 0]))
        self.dof = self.k - 1
        self.H = self._hvalue()
        self.p_value = self._pvalue()
        self.t_value = self._tvalue()
        self.least_significant_difference = self._lsd()
        self.test_description = 'Kruskal-Wallis rank sum test'

    def _hvalue(self):
        group_observations = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1:], len)

        group_observations = np.array([i for _, i in group_observations])

        group_summed_ranks = np.array([i for _, i in self.group_rank_sums])

        h1 = 12. / (self.n * (self.n + 1))
        h2 = np.sum(group_summed_ranks ** 2 / group_observations)

        h = h1 * h2 - (3 * (self.n + 1))

        # Apply tie correction if needed
        if len(np.unique(self.ranked_matrix[:, 2])) != self.n:

            h /= tie_correction(self.ranked_matrix[:, 2])

        return h

    def _pvalue(self):
        p = 1 - chi2.cdf(self.H, self.dof)

        return p

    def _tvalue(self):
        tval = t.ppf(1 - self.alpha / 2, self.n - self.k)

        return tval

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

    def _lsd(self):
        lsd = self.t_value * np.sqrt(self._mse() * 2 / (self.n / self.k))

        return lsd

    def summary(self):
        test_results = {'test description': self.test_description,
                        'critical chisq value': self.H,
                        'p-value': self.p_value,
                        'least significant difference': self.least_significant_difference,
                        't-value': self.t_value,
                        'alpha': self.alpha,
                        'degrees of freedom': self.dof
        }

        return test_results


def tie_correction(rank_array):
    tied_groups = np.unique(rank_array, return_counts=True)[1]
    tied_groups = tied_groups[tied_groups > 1]

    h = 1 - np.sum((tied_groups ** 3 - tied_groups)) / (rank_array.shape[0] ** 3 -
                                                        rank_array.shape[0])

    return h
