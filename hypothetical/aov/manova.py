import numpy as np
import numpy_indexed as npi

from hypothetical._lib import build_des_mat


class ManovaOneWay(object):

    def __init__(self, *args, group):

        self.design_matrix = build_des_mat(*args, group=group)

        if group is not None:
            self.group = group
        else:
            self.group = self.design_matrix[:, 0]


        self.design_matrix = build_des_mat(group, *args)
        #self.group_names = np.unique(self.group)
        self.k = len(np.unique(self.group))

        self._group_stats = self._group_statistics()
        self._observation_stats = self._obs_statistics()

        self.hypothesis_matrix, self.error_matrix = self._hypothesis_error_matrix()
        self._intermediate_statistic_parameters = self._intermediate_test_statistic_parameters()

        #self.degrees_of_freedom = self._degrees_of_freedom()
        #self.pillai_statistic = self.pillai_statistic()
        #self.wilks_lambda = self._wilks_statistic()
        #self.roys_statistic = self._roys_statistic()
        #self.hotelling_t2_statistic = self._hotelling_t2_statistic()

    def _group_statistics(self):
        group_means = npi.group_by(self.design_matrix[:, 0]).mean(self.design_matrix[:, 1])[1] #.mean(self.design_matrix[:, 1:])[1][:, 1:]

        group_observations = npi.group_by(self.design_matrix[:, 0], self.design_matrix[:, 1:], len)
        group_observations = [i for _, i in group_observations]

        groups = npi.group_by(self.design_matrix[:, 0], self.design_matrix)[1]

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

        xn = self._observation_stats['x observations']
        n = self._group_stats['Group Observations']

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
        n = len(self.x)

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
                       'Pillai F-value': pillai_f}

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

        wilks_stat = {"Wilk's Lambda": wilks_lambda,
                      "Wilk's Lambda F-value": wilks_lambda_f}

        return wilks_stat

    def _roys_statistic(self):
        eigs, n, vh = self._intermediate_statistic_parameters['eigs'], \
                      self._intermediate_statistic_parameters['n'], \
                      self._intermediate_statistic_parameters['vh']

        roy = np.max(eigs)
        roy_f = float(self.k * (n - 1)) / float(vh) * roy

        roy_stat = {"Roy's Statistic": roy,
                    "Roy F-value": roy_f}

        return roy_stat

    def _hotelling_t2_statistic(self):
        dot_inve_h, s, nn, m = self._intermediate_statistic_parameters['dot_inve_h'], \
                               self._intermediate_statistic_parameters['s'], \
                               self._intermediate_statistic_parameters['nn'], \
                               self._intermediate_statistic_parameters['m']

        t2 = np.sum(np.diag(dot_inve_h))
        t2_f = (2. * (s * nn + 1.) * np.sum(dot_inve_h)) / (s ** 2. * (2. * m + s + 1.))

        t2_stat = {
            "Hotelling's T^2 Statistic": t2,
            "Hotelling's T^2 F-value": t2_f
        }

        return t2_stat

    def _degrees_of_freedom(self):
        vh, ve, xn = self._intermediate_statistic_parameters['vh'], \
                     self._intermediate_statistic_parameters['ve'], \
                     self._intermediate_statistic_parameters['xn']

        num_df, denom_df = vh * xn, ve * xn

        dof = {'Numerator Degrees of Freedom': num_df,
               'Denominator Degrees of Freedom': denom_df}

        return dof

    def _f_p_value(self, f, df_num, df_denom):
        p = 1 - f.cdf(f, df_num, df_denom)

        return p

    @staticmethod
    def _dot_inve_h(h, e):
        return np.dot(np.linalg.inv(e), h)