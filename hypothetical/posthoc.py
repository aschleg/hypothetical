from hypothetical._lib import build_des_mat
import numpy as np
import pandas as pd
import numpy_indexed as npi
from hypothetical.summary import var, std_dev
from statsmodels.stats.libqsturng import qsturng, psturng
from itertools import combinations


class TukeysTest(object):

    def __init__(self, *args, group=None, alpha=0.95):
        self.alpha = alpha
        self.test_description = 'Tukey multiple comparisons of means'

        self.design_matrix = build_des_mat(*args, group=group)

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

    def _mse(self):

        group_variance = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], var)
        group_n = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1], len)

        sse = 0

        for i, j in zip(group_n, group_variance):
            sse += (i[1] - 1) * j[1]

        return sse / (self.n - self.k)

    def _qvalue(self):
        q = qsturng(self.alpha, self.k, self.n - self.k)

        return q

    def _hsd(self):
        hsd = self.tukey_q_value * np.sqrt(self.mse / (self.n / self.k))

        return hsd

    def _group_comparison(self):
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

        groups['upper interval'] = groups['mean difference'] + \
                                   self.tukey_q_value * np.sqrt(self.mse / 2. * (2. / (self.n / self.k)))

        groups['lower interval'] = groups['mean difference'] - \
                                   self.tukey_q_value * np.sqrt(self.mse / 2. * (2. / (self.n / self.k)))

        q_values = groups['mean difference'] / group_sd

        groups['p_adjusted'] = psturng(np.absolute(q_values), self.n / self.k, self.dof)

        del groups[0]
        del groups[1]

        return groups

    def summary(self):
        test_results = {
            'test description': self.test_description,
            'HSD': self.hsd,
            'MSE': self.mse,
            'Studentized Range q-value': self.tukey_q_value,
            'degrees of freedom': self.dof,
            'group comparisons': self.group_comparison.to_dict(),
            'alpha': self.alpha
        }

        return test_results


class GamesHowell(object):

    def __init__(self):
        pass
