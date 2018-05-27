import numpy as np
from scipy.stats import rankdata, norm

from hypothetical.nonparametric.mann_whitney import MannWhitney


def wilcox_test(y1, y2=None, paired=False, median=0, continuity=True):
    if y2 is not None and paired is False:
        res = MannWhitney(y1=y1, y2=y2, continuity=continuity)
    else:
        res = WilcoxonTest(y1=y1, y2=y2, paired=paired, median=median, continuity=continuity)

    return res


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