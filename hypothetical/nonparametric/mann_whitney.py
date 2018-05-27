import numpy as np
import numpy_indexed as npi
from scipy.stats import rankdata, norm

from hypothetical.nonparametric import wilcoxon


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
        res = wilcoxon.WilcoxonTest(y1=y1)
    else:
        res = MannWhitney(y1=y1, y2=y2, group=group, continuity=continuity)

    return res


class MannWhitney(object):

    def __init__(self, y1, y2=None, group=None, continuity=True):

        if group is None:
            self.y1 = y1
            self.y2 = y2
        else:
            if len(group.unique()) > 2:
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
