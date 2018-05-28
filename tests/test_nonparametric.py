import pytest

from hypothetical.nonparametric import mann_whitney, wilcox_test
import pandas as pd
import numpy as np
import os


@pytest.fixture
def test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    salaries = pd.read_csv(os.path.join(datapath, '../data/Salaries.csv'))

    return salaries


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
    w = wilcox_test(sal_a).summary()

    assert mw2 == w


def test_wilcox_test(test_data):
    sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
    sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

    w = wilcox_test(sal_a)

    test_result = w.summary()

    np.testing.assert_equal(test_result['V'], 16471.0)
    np.testing.assert_almost_equal(test_result['p-value'], np.finfo(float).eps)
    np.testing.assert_almost_equal(test_result['z-value'], 11.667217617844829)

    mw = mann_whitney(sal_a, sal_b).summary()
    w2 = wilcox_test(sal_a, sal_b).summary()

    assert mw == w2

    assert test_result['test description'] == 'Wilcoxon signed rank test'

    with pytest.raises(ValueError):
        wilcox_test(sal_a, sal_b, paired=True)
    with pytest.raises(ValueError):
        wilcox_test(sal_a, paired=True)
