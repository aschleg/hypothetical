import pytest

from hypothetical.nonparametric import mann_whitney, wilcoxon_test, tie_correction, kruskal_wallis
import pandas as pd
import numpy as np
import os
from scipy.stats import rankdata, tiecorrect


@pytest.fixture
def test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    salaries = pd.read_csv(os.path.join(datapath, '../data/Salaries.csv'))

    return salaries


@pytest.fixture
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


@pytest.fixture
def plants_test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    plants = pd.read_csv(os.path.join(datapath, '../data/PlantGrowth.csv'))

    return plants


def test_mann_whitney(test_data):
    sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
    sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

    mw = mann_whitney(sal_a, sal_b)

    test_result = mw.summary()

    np.testing.assert_equal(test_result['U'], 15710.0)
    np.testing.assert_almost_equal(test_result['p-value'], 0.0007492490583558276)
    np.testing.assert_almost_equal(test_result['mu meanrank'], 19548.5)
    np.testing.assert_almost_equal(test_result['sigma'], 1138.718969482228)
    np.testing.assert_almost_equal(test_result['z-value'], 3.3708931728303027)

    assert test_result['continuity']

    mw2 = mann_whitney(group=test_data['discipline'], y1=test_data['salary'])

    assert test_result == mw2.summary()

    mw_no_cont = mann_whitney(sal_a, sal_b, continuity=False)

    no_cont_result = mw_no_cont.summary()

    np.testing.assert_equal(no_cont_result['U'], 15710.0)
    np.testing.assert_almost_equal(no_cont_result['p-value'], 0.0007504441137706763)
    np.testing.assert_almost_equal(no_cont_result['mu meanrank'], 19548.0)
    np.testing.assert_almost_equal(no_cont_result['sigma'], 1138.718969482228)
    np.testing.assert_almost_equal(no_cont_result['z-value'], 3.370454082928931)

    assert no_cont_result['continuity'] is False

    mw2 = mann_whitney(sal_a).summary()
    w = wilcoxon_test(sal_a).summary()

    assert mw2 == w

    mult_data = multivariate_test_data()

    with pytest.raises(ValueError):
        mann_whitney(y1=mult_data[:, 1], group=mult_data[:, 0])


def test_wilcox_test(test_data):
    mult_data = multivariate_test_data()

    sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
    sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

    w = wilcoxon_test(sal_a)

    test_result = w.summary()

    np.testing.assert_equal(test_result['V'], 16471.0)
    np.testing.assert_almost_equal(test_result['p-value'], np.finfo(float).eps)
    np.testing.assert_almost_equal(test_result['z-value'], 11.667217617844829)

    mw = mann_whitney(sal_a, sal_b).summary()
    w2 = wilcoxon_test(sal_a, sal_b).summary()

    assert mw == w2

    assert test_result['test description'] == 'Wilcoxon signed rank test'

    with pytest.raises(ValueError):
        wilcoxon_test(sal_a, sal_b, paired=True)
    with pytest.raises(ValueError):
        wilcoxon_test(sal_a, paired=True)

    paired_w = wilcoxon_test(mult_data[:, 1], mult_data[:, 2], paired=True)

    paired_result = paired_w.summary()

    np.testing.assert_equal(paired_result['V'], 1176.0)
    np.testing.assert_almost_equal(paired_result['p-value'], 1.6310099937300038e-09)
    np.testing.assert_almost_equal(paired_result['z-value'], 6.030848532388999)

    assert paired_result['test description'] == 'Wilcoxon signed rank test'


def test_kruskal_wallis():
    data = plants_test_data()

    kw = kruskal_wallis(data['weight'], group=data['group'])

    test_result = kw.summary()

    assert test_result['alpha'] == 0.05
    assert test_result['degrees of freedom'] == 2
    assert test_result['test description'] == 'Kruskal-Wallis rank sum test'

    np.testing.assert_almost_equal(test_result['critical chisq value'], 7.988228749443715)
    np.testing.assert_almost_equal(test_result['least significant difference'], 7.125387208146856)
    np.testing.assert_almost_equal(test_result['p-value'], 0.018423755731471925)
    np.testing.assert_almost_equal(test_result['t-value'], 2.0518305164802833)

    del data['Unnamed: 0']

    ctrl = data[data['group'] == 'ctrl']['weight'].reset_index()
    del ctrl['index']
    ctrl.rename(columns={'weight': 'ctrl'}, inplace=True)

    trt1 = data[data['group'] == 'trt1']['weight'].reset_index()

    del trt1['index']
    trt1.rename(columns={'weight': 'trt1'}, inplace=True)

    trt2 = data[data['group'] == 'trt2']['weight'].reset_index()

    del trt2['index']
    trt2.rename(columns={'weight': 'trt2'}, inplace=True)

    kw2 = kruskal_wallis(ctrl, trt1, trt2)

    test_result2 = kw2.summary()

    assert test_result == test_result2


def test_tie_correction():
    mult_data = multivariate_test_data()

    ranks = rankdata(mult_data[:, 1], 'average')

    ranks = np.column_stack([mult_data, ranks])

    tie_correct = tie_correction(ranks[:, 5])

    np.testing.assert_almost_equal(tie_correct, tiecorrect(ranks[:, 5]))
