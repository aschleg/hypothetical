# encoding=utf-8

from collections import namedtuple

import numpy as np
from scipy.stats import rankdata, norm


def mann_whitney(y1, y2, continuity=True):
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
    n1, n2 = len(y1), len(y2)

    ranks = np.concatenate((y1, y2))

    ranks = rankdata(ranks, 'average')

    ranks = ranks[:n1]

    n = n1 + n2

    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2. - np.sum(ranks)
    u2 = n1 * n2 - u1

    u = np.minimum(u1, u2)
    mu = (n1 * n2) / 2. + (0.5 * continuity)

    rankcounts = np.unique(ranks, return_counts=True)[1]

    sigma = np.sqrt(((n1 * n2) * (n + 1)) / 12. * (1 - np.sum(rankcounts ** 3 - rankcounts) / float(n ** 3 - n)))
    z = (np.absolute(u - mu)) / sigma
    p = 1-norm.cdf(z)

    MannWhitneyResult = namedtuple('MannWhitneyResult', ['u', 'meanrank', 'sigma', 'zvalue', 'pvalue'])

    mwr = MannWhitneyResult(u=u, meanrank=mu, sigma=sigma, zvalue=z, pvalue=p)

    return mwr