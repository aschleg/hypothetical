import pytest
import hypy
import pandas as pd
import numpy as np
import os
from scipy.stats import t


@pytest.fixture
def test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    salaries = pd.read_csv(os.path.join(datapath, 'test_data/Salaries.csv'))

    return salaries


@pytest.fixture
def test_multiclass_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    insectsprays = pd.read_csv(os.path.join(datapath, 'test_data/InsectSprays.csv'))

    return insectsprays


def test_two_sample_welch_test(test_data):
    sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
    sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

    ttest = hypy.t_test(y1=sal_a, y2=sal_b)

    test_summary = ttest.summary()

    np.testing.assert_almost_equal(test_summary['Sample 1 Mean'], np.mean(sal_a))
    np.testing.assert_almost_equal(test_summary['Sample 2 Mean'], np.mean(sal_b))
    np.testing.assert_almost_equal(test_summary['t-statistic'], -3.1386989278486013)
    np.testing.assert_almost_equal(test_summary['degrees of freedom'], 377.89897288941387)
    np.testing.assert_almost_equal(test_summary['p-value'], t.cdf(test_summary['t-statistic'],
                                                                  test_summary['degrees of freedom']) * 2)

    assert test_summary['alternative'] == 'two-sided'
    assert test_summary['test description'] == "Two-Sample Welch's t-test"

    ttest_group = hypy.t_test(group=test_data['discipline'], y1=test_data['salary'])
    test_group_summary = ttest_group.summary()

    np.testing.assert_almost_equal(test_summary['Sample 1 Mean'], test_group_summary['Sample 1 Mean'])
    np.testing.assert_almost_equal(test_summary['Sample 2 Mean'], test_group_summary['Sample 2 Mean'])
    np.testing.assert_almost_equal(test_summary['p-value'], test_group_summary['p-value'])
    np.testing.assert_almost_equal(test_summary['degrees of freedom'], test_group_summary['degrees of freedom'], 5)
    np.testing.assert_almost_equal(test_summary['t-statistic'], test_group_summary['t-statistic'])

    assert test_group_summary['alternative'] == 'two-sided'
    assert test_group_summary['test description'] == "Two-Sample Welch's t-test"


def test_two_sample_students_test(test_data):
    sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
    sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

    ttest = hypy.t_test(y1=sal_a, y2=sal_b, var_equal=True)

    test_summary = ttest.summary()

    np.testing.assert_almost_equal(test_summary['Sample 1 Mean'], np.mean(sal_a))
    np.testing.assert_almost_equal(test_summary['Sample 2 Mean'], np.mean(sal_b))
    np.testing.assert_almost_equal(test_summary['t-statistic'], -3.1485647713976195)
    np.testing.assert_almost_equal(test_summary['p-value'], t.cdf(test_summary['t-statistic'],
                                                                  test_summary['degrees of freedom']) * 2)

    assert test_summary['alternative'] == 'two-sided'
    assert test_summary['test description'] == "Two-Sample Student's t-test"

    assert len(sal_a) + len(sal_b) - 2 == test_summary['degrees of freedom']


def test_one_sample_test(test_data):
    sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']

    ttest = hypy.t_test(y1=sal_a)

    test_summary = ttest.summary()

    np.testing.assert_almost_equal(test_summary['Sample 1 Mean'], np.mean(sal_a))
    np.testing.assert_almost_equal(test_summary['t-statistic'], 47.95382017797468)
    np.testing.assert_almost_equal(test_summary['p-value'], 2.220446049250313e-16)

    assert test_summary['alternative'] == 'two-sided'
    assert test_summary['test description'] == 'One-Sample t-test'

    assert len(sal_a) - 1 == test_summary['degrees of freedom']

    ttest_mu = hypy.t_test(y1=sal_a, mu=100000)

    test_mu_summary = ttest_mu.summary()

    np.testing.assert_almost_equal(test_mu_summary['Sample 1 Mean'], np.mean(sal_a))
    np.testing.assert_almost_equal(test_mu_summary['p-value'], 0.0002159346891279501)
    np.testing.assert_almost_equal(test_mu_summary['t-statistic'], 3.776470249422699)

    assert test_mu_summary['alternative'] == 'two-sided'
    assert test_mu_summary['test description'] == 'One-Sample t-test'
    assert test_mu_summary['mu'] == 100000
    assert len(sal_a) - 1 == test_mu_summary['degrees of freedom']


def test_paired_sample_test(test_data):
    sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
    sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']
    sal_b2 = sal_b[0:len(sal_a)]

    ttest = hypy.t_test(y1=sal_a, y2=sal_b2, paired=True)

    test_summary = ttest.summary()

    np.testing.assert_almost_equal(test_summary['Sample Difference Mean'], np.mean(np.array(sal_a) - np.array(sal_b2)))
    np.testing.assert_almost_equal(test_summary['t-statistic'], -2.3158121700626406)
    np.testing.assert_almost_equal(test_summary['p-value'], t.cdf(test_summary['t-statistic'],
                                                                  test_summary['degrees of freedom']) * 2)

    assert test_summary['alternative'] == 'two-sided'
    assert test_summary['test description'] == 'Paired t-test'

    assert len(sal_a) - 1 == test_summary['degrees of freedom']


def test_alternatives(test_data):
    sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
    sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

    ttest = hypy.t_test(y1=sal_a, y2=sal_b, alternative='greater')

    test_summary = ttest.summary()

    np.testing.assert_almost_equal(test_summary['p-value'], 0.9990848459959981)
    np.testing.assert_almost_equal(test_summary['t-statistic'], -3.1386989278486013)

    assert test_summary['alternative'] == 'greater'


def test_ttest_exceptions(test_data, test_multiclass_data):
    sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
    sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

    with pytest.raises(ValueError):
        hypy.t_test(y1=sal_a, paired=True)

    with pytest.raises(ValueError):
        hypy.t_test(y1=sal_a, y2=sal_b, paired=True)

    with pytest.raises(ValueError):
        hypy.t_test(sal_a, sal_b, alternative='asdh')

    with pytest.raises(ValueError):
        hypy.t_test(group=test_multiclass_data['spray'], y1=test_multiclass_data['count'])
