import pytest

from hypothetical.nonparametric import KruskalWallis, MannWhitney, SignTest, tie_correction, WilcoxonTest, MedianTest
import pandas as pd
import numpy as np
import os
from scipy.stats import rankdata, tiecorrect


def test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    salaries = pd.read_csv(os.path.join(datapath, 'data/Salaries.csv'))

    return salaries


def multivariate_test_data():
    d = np.array([[1., 1.11, 2.569, 3.58, 0.76],
                  [1., 1.19, 2.928, 3.75, 0.821],
                  [1., 1.09, 2.865, 3.93, 0.928],
                  [1., 1.25, 3.844, 3.94, 1.009],
                  [1., 1.11, 3.027, 3.6, 0.766],
                  [1., 1.08, 2.336, 3.51, 0.726],
                  [1., 1.11, 3.211, 3.98, 1.209],
                  [1., 1.16, 3.037, 3.62, 0.75],
                  [2., 1.05, 2.074, 4.09, 1.036],
                  [2., 1.17, 2.885, 4.06, 1.094],
                  [2., 1.11, 3.378, 4.87, 1.635],
                  [2., 1.25, 3.906, 4.98, 1.517],
                  [2., 1.17, 2.782, 4.38, 1.197],
                  [2., 1.15, 3.018, 4.65, 1.244],
                  [2., 1.17, 3.383, 4.69, 1.495],
                  [2., 1.19, 3.447, 4.4, 1.026],
                  [3., 1.07, 2.505, 3.76, 0.912],
                  [3., 0.99, 2.315, 4.44, 1.398],
                  [3., 1.06, 2.667, 4.38, 1.197],
                  [3., 1.02, 2.39, 4.67, 1.613],
                  [3., 1.15, 3.021, 4.48, 1.476],
                  [3., 1.2, 3.085, 4.78, 1.571],
                  [3., 1.2, 3.308, 4.57, 1.506],
                  [3., 1.17, 3.231, 4.56, 1.458],
                  [4., 1.22, 2.838, 3.89, 0.944],
                  [4., 1.03, 2.351, 4.05, 1.241],
                  [4., 1.14, 3.001, 4.05, 1.023],
                  [4., 1.01, 2.439, 3.92, 1.067],
                  [4., 0.99, 2.199, 3.27, 0.693],
                  [4., 1.11, 3.318, 3.95, 1.085],
                  [4., 1.2, 3.601, 4.27, 1.242],
                  [4., 1.08, 3.291, 3.85, 1.017],
                  [5., 0.91, 1.532, 4.04, 1.084],
                  [5., 1.15, 2.552, 4.16, 1.151],
                  [5., 1.14, 3.083, 4.79, 1.381],
                  [5., 1.05, 2.33, 4.42, 1.242],
                  [5., 0.99, 2.079, 3.47, 0.673],
                  [5., 1.22, 3.366, 4.41, 1.137],
                  [5., 1.05, 2.416, 4.64, 1.455],
                  [5., 1.13, 3.1, 4.57, 1.325],
                  [6., 1.11, 2.813, 3.76, 0.8],
                  [6., 0.75, 0.84, 3.14, 0.606],
                  [6., 1.05, 2.199, 3.75, 0.79],
                  [6., 1.02, 2.132, 3.99, 0.853],
                  [6., 1.05, 1.949, 3.34, 0.61],
                  [6., 1.07, 2.251, 3.21, 0.562],
                  [6., 1.13, 3.064, 3.63, 0.707],
                  [6., 1.11, 2.469, 3.95, 0.952]])

    return d


def plants_test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    plants = pd.read_csv(os.path.join(datapath, 'data/PlantGrowth.csv'))

    return plants


class TestMannWhitney(object):

    data = test_data()
    mult_data = multivariate_test_data()

    def test_mann_whitney(self):
        sal_a = self.data.loc[self.data['discipline'] == 'A']['salary']
        sal_b = self.data.loc[self.data['discipline'] == 'B']['salary']

        mw = MannWhitney(sal_a, sal_b)

        test_result = mw.test_summary

        np.testing.assert_equal(test_result['U'], 15710.0)
        np.testing.assert_almost_equal(test_result['p-value'], 0.0007492490583558276)
        np.testing.assert_almost_equal(test_result['mu meanrank'], 19548.5)
        np.testing.assert_almost_equal(test_result['sigma'], 1138.718969482228)
        np.testing.assert_almost_equal(test_result['z-value'], 3.3708931728303027)

        assert test_result['continuity']

        mw2 = MannWhitney(group=self.data['discipline'],
                          y1=self.data['salary'])

        assert test_result == mw2.test_summary

        mw_no_cont = MannWhitney(sal_a, sal_b, continuity=False)

        no_cont_result = mw_no_cont.test_summary

        np.testing.assert_equal(no_cont_result['U'], 15710.0)
        np.testing.assert_almost_equal(no_cont_result['p-value'], 0.0007504441137706763)
        np.testing.assert_almost_equal(no_cont_result['mu meanrank'], 19548.0)
        np.testing.assert_almost_equal(no_cont_result['sigma'], 1138.718969482228)
        np.testing.assert_almost_equal(no_cont_result['z-value'], 3.370454082928931)

        assert no_cont_result['continuity'] is False

    def test_exceptions(self):
        with pytest.raises(ValueError):
            MannWhitney(y1=self.mult_data[:, 1],
                        group=self.mult_data[:, 0])


class TestWilcoxon(object):

    data = test_data()
    mult_data = multivariate_test_data()

    sal_a = data.loc[data['discipline'] == 'A']['salary']
    sal_b = data.loc[data['discipline'] == 'B']['salary']

    def test_wilcoxon_one_sample(self):
        w = WilcoxonTest(self.sal_a)

        test_result = w.test_summary

        np.testing.assert_equal(test_result['V'], 16471.0)
        np.testing.assert_almost_equal(test_result['p-value'], np.finfo(float).eps)
        np.testing.assert_almost_equal(test_result['z-value'], 11.667217617844829)

    def test_wilcoxon_multi_sample(self):
        paired_w = WilcoxonTest(self.mult_data[:, 1], self.mult_data[:, 2], paired=True)

        paired_result = paired_w.test_summary

        np.testing.assert_equal(paired_result['V'], 1176.0)
        np.testing.assert_almost_equal(paired_result['p-value'], 1.6310099937300038e-09)
        np.testing.assert_almost_equal(paired_result['z-value'], 6.030848532388999)

        assert paired_result['test description'] == 'Wilcoxon signed rank test'

    def test_exceptions(self):
        with pytest.raises(ValueError):
            WilcoxonTest(self.sal_a, self.sal_b, paired=True)


class TestKruskalWallis(object):
    data = plants_test_data()

    def test_kruskal_wallis(self):
        kw = KruskalWallis(self.data['weight'], group=self.data['group'])

        test_result = kw.test_summary

        assert test_result['alpha'] == 0.05
        assert test_result['degrees of freedom'] == 2
        assert test_result['test description'] == 'Kruskal-Wallis rank sum test'

        np.testing.assert_almost_equal(test_result['critical chisq value'], 7.988228749443715)
        np.testing.assert_almost_equal(test_result['least significant difference'], 7.125387208146856)
        np.testing.assert_almost_equal(test_result['p-value'], 0.018423755731471925)
        np.testing.assert_almost_equal(test_result['t-value'], 2.0518305164802833)

        del self.data['Unnamed: 0']

        ctrl = self.data[self.data['group'] == 'ctrl']['weight'].reset_index()
        del ctrl['index']
        ctrl.rename(columns={'weight': 'ctrl'}, inplace=True)

        trt1 = self.data[self.data['group'] == 'trt1']['weight'].reset_index()

        del trt1['index']
        trt1.rename(columns={'weight': 'trt1'}, inplace=True)

        trt2 = self.data[self.data['group'] == 'trt2']['weight'].reset_index()

        del trt2['index']
        trt2.rename(columns={'weight': 'trt2'}, inplace=True)

        kw2 = KruskalWallis(ctrl, trt1, trt2)

        test_result2 = kw2.test_summary

        assert test_result == test_result2

    def test_exceptions(self):
        with pytest.raises(ValueError):
            KruskalWallis(self.data['weight'],
                          self.data['weight'],
                          group=self.data['group'])


class TestSignTest(object):
    f = [4, 4, 5, 5, 3, 2, 5, 3, 1, 5, 5, 5, 4, 5, 5, 5, 5]
    m = [2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 5, 2, 5, 3, 1]

    fm = [[4, 4, 5, 5, 3, 2, 5, 3, 1, 5, 5, 5, 4, 5, 5, 5, 5],
          [2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 5, 2, 5, 3, 1]]

    fmm = [[4, 4, 5, 5, 3, 2, 5, 3, 1, 5, 5, 5, 4, 5, 5, 5, 5],
           [2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 5, 2, 5, 3, 1],
           [2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 5, 2, 5, 3, 1]]

    def test_sign_test(self):
        s = SignTest(self.f, self.m)

        np.testing.assert_almost_equal(s.p_value, 0.057373046875)
        assert s.differences_counts['positive'] == 11
        assert s.differences_counts['negative'] == 3
        assert s.differences_counts['ties'] == 3
        assert s.sample_differences_median == 2
        assert s.alternative == 'two-sided'

        s2 = SignTest(np.array(self.f), np.array(self.m))

        np.testing.assert_almost_equal(s.p_value, s2.p_value)

    def test_sign_test_less(self):
        s = SignTest(self.f, self.m, alternative='less')

        np.testing.assert_almost_equal(s.p_value, 0.9935302734375)

    def test_sign_test_greater(self):
        s = SignTest(self.f, self.m, alternative='greater')

        np.testing.assert_almost_equal(s.p_value, 0.0286865234375)

    def test_sign_test_exceptions(self):
        with pytest.raises(ValueError):
            SignTest(self.f[0:5], self.m)

        with pytest.raises(ValueError):
            SignTest(self.fmm)

        with pytest.raises(ValueError):
            SignTest(self.f)

        with pytest.raises(ValueError):
            SignTest(self.f, self.m, alternative='na')


class TestMedianTest(object):
    g1 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
    g2 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
    g3 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]

    def test_mediantest(self):
        m = MedianTest(self.g1, self.g2, self.g3)

        np.testing.assert_almost_equal(m.test_statistic, 4.141505553270259)
        np.testing.assert_almost_equal(m.p_value, 0.12609082774093244)
        np.testing.assert_almost_equal(m.grand_median, 34.0)
        np.testing.assert_array_almost_equal(m.contingency_table, np.array([[5, 10,  7], [11,  5, 10]]))

    def test_median_ties_above(self):
        m = MedianTest(self.g1, self.g2, self.g3, ties='above')

        np.testing.assert_almost_equal(m.test_statistic, 5.5017084398976985)
        np.testing.assert_almost_equal(m.p_value, 0.06387327606955327)
        np.testing.assert_array_almost_equal(m.contingency_table, np.array([[5, 11,  9], [11,  4,  8]]))

    def test_median_ties_ignore(self):
        m = MedianTest(self.g1, self.g2, self.g3, ties='ignore')

        np.testing.assert_almost_equal(m.test_statistic, 4.868277103331452)
        np.testing.assert_almost_equal(m.p_value, 0.08767324049352121)
        np.testing.assert_array_almost_equal(m.contingency_table, np.array([[5, 10,  7], [11,  4,  8]]))

    def test_median_continuity(self):
        m = MedianTest(self.g1, self.g2)

        np.testing.assert_almost_equal(m.test_statistic, 2.5996137152777785)
        np.testing.assert_almost_equal(m.p_value, 0.10688976489998428)
        np.testing.assert_almost_equal(m.contingency_table, np.array([[5, 10], [11,  5]]))

    def test_median_no_continuity(self):
        m = MedianTest(self.g1, self.g2, continuity=False)

        np.testing.assert_almost_equal(m.test_statistic, 3.888454861111112)
        np.testing.assert_almost_equal(m.p_value, 0.04861913422927604)
        np.testing.assert_almost_equal(m.contingency_table, np.array([[5, 10], [11,  5]]))

    def test_median_exceptions(self):
        with pytest.raises(ValueError):
            MedianTest(self.g1, self.g2, ties='na')


def test_tie_correction():
    mult_data = multivariate_test_data()

    ranks = rankdata(mult_data[:, 1], 'average')

    ranks = np.column_stack([mult_data, ranks])

    tie_correct = tie_correction(ranks[:, 5])

    np.testing.assert_almost_equal(tie_correct, tiecorrect(ranks[:, 5]))
