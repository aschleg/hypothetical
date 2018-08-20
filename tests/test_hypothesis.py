

import pytest
from hypothetical.hypothesis import tTest, BinomialTest
import pandas as pd
import numpy as np
import os
from scipy.stats import t


@pytest.fixture
def test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    salaries = pd.read_csv(os.path.join(datapath, '../data/Salaries.csv'))

    return salaries


@pytest.fixture
def test_multiclass_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    insectsprays = pd.read_csv(os.path.join(datapath, '../data/InsectSprays.csv'))

    return insectsprays


class Test_tTest(object):

    def test_two_sample_welch_test(self, test_data):
        sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
        sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

        ttest = tTest(y1=sal_a, y2=sal_b)

        test_summary = ttest.test_summary

        np.testing.assert_almost_equal(test_summary['Sample 1 Mean'], np.mean(sal_a))
        np.testing.assert_almost_equal(test_summary['Sample 2 Mean'], np.mean(sal_b))
        np.testing.assert_almost_equal(test_summary['t-statistic'], -3.1386989278486013)
        np.testing.assert_almost_equal(test_summary['degrees of freedom'], 377.89897288941387)
        np.testing.assert_almost_equal(test_summary['p-value'], t.cdf(test_summary['t-statistic'],
                                                                      test_summary['degrees of freedom']) * 2)

        assert test_summary['alternative'] == 'two-sided'
        assert test_summary['test description'] == "Two-Sample Welch's t-test"

        ttest_group = tTest(group=test_data['discipline'], y1=test_data['salary'])
        test_group_summary = ttest_group.test_summary

        np.testing.assert_almost_equal(test_summary['Sample 1 Mean'], test_group_summary['Sample 1 Mean'])
        np.testing.assert_almost_equal(test_summary['Sample 2 Mean'], test_group_summary['Sample 2 Mean'])
        np.testing.assert_almost_equal(test_summary['p-value'], test_group_summary['p-value'])
        np.testing.assert_almost_equal(test_summary['degrees of freedom'], test_group_summary['degrees of freedom'], 5)
        np.testing.assert_almost_equal(test_summary['t-statistic'], test_group_summary['t-statistic'])

        assert test_group_summary['alternative'] == 'two-sided'
        assert test_group_summary['test description'] == "Two-Sample Welch's t-test"

    def test_two_sample_students_test(self, test_data):
        sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
        sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

        ttest = tTest(y1=sal_a, y2=sal_b, var_equal=True)

        test_summary = ttest.test_summary

        np.testing.assert_almost_equal(test_summary['Sample 1 Mean'], np.mean(sal_a))
        np.testing.assert_almost_equal(test_summary['Sample 2 Mean'], np.mean(sal_b))
        np.testing.assert_almost_equal(test_summary['t-statistic'], -3.1485647713976195)
        np.testing.assert_almost_equal(test_summary['p-value'], t.cdf(test_summary['t-statistic'],
                                                                      test_summary['degrees of freedom']) * 2)

        assert test_summary['alternative'] == 'two-sided'
        assert test_summary['test description'] == "Two-Sample Student's t-test"

        assert len(sal_a) + len(sal_b) - 2 == test_summary['degrees of freedom']

    def test_one_sample_test(self, test_data):
        sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']

        ttest = tTest(y1=sal_a)

        test_summary = ttest.test_summary

        np.testing.assert_almost_equal(test_summary['Sample 1 Mean'], np.mean(sal_a))
        np.testing.assert_almost_equal(test_summary['t-statistic'], 47.95382017797468)
        np.testing.assert_almost_equal(test_summary['p-value'], 2.220446049250313e-16)

        assert test_summary['alternative'] == 'two-sided'
        assert test_summary['test description'] == 'One-Sample t-test'

        assert len(sal_a) - 1 == test_summary['degrees of freedom']

        ttest_mu = tTest(y1=sal_a, mu=100000)

        test_mu_summary = ttest_mu._generate_result_summary()

        np.testing.assert_almost_equal(test_mu_summary['Sample 1 Mean'], np.mean(sal_a))
        np.testing.assert_almost_equal(test_mu_summary['p-value'], 0.0002159346891279501)
        np.testing.assert_almost_equal(test_mu_summary['t-statistic'], 3.776470249422699)

        assert test_mu_summary['alternative'] == 'two-sided'
        assert test_mu_summary['test description'] == 'One-Sample t-test'
        assert test_mu_summary['mu'] == 100000
        assert len(sal_a) - 1 == test_mu_summary['degrees of freedom']

    def test_paired_sample_test(self, test_data):
        sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
        sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']
        sal_b2 = sal_b[0:len(sal_a)]

        ttest = tTest(y1=sal_a, y2=sal_b2, paired=True)

        test_summary = ttest.test_summary

        np.testing.assert_almost_equal(test_summary['Sample Difference Mean'], np.mean(np.array(sal_a) - np.array(sal_b2)))
        np.testing.assert_almost_equal(test_summary['t-statistic'], -2.3158121700626406)
        np.testing.assert_almost_equal(test_summary['p-value'], t.cdf(test_summary['t-statistic'],
                                                                      test_summary['degrees of freedom']) * 2)

        assert test_summary['alternative'] == 'two-sided'
        assert test_summary['test description'] == 'Paired t-test'

        assert len(sal_a) - 1 == test_summary['degrees of freedom']

    def test_alternatives(self, test_data):
        sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
        sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

        ttest = tTest(y1=sal_a, y2=sal_b, alternative='greater')

        test_summary = ttest.test_summary

        np.testing.assert_almost_equal(test_summary['p-value'], 0.9990848459959981)
        np.testing.assert_almost_equal(test_summary['t-statistic'], -3.1386989278486013)

        assert test_summary['alternative'] == 'greater'

        ttest_less = tTest(y1=sal_a, y2=sal_b, alternative='less')

        test_less_summary = ttest_less.test_summary

        assert test_less_summary['alternative'] == 'less'
        np.testing.assert_almost_equal(test_less_summary['t-statistic'], -3.1386989278486013)
        np.testing.assert_almost_equal(test_less_summary['p-value'], 0.0009151540040019292)

    def test_ttest_exceptions(self, test_data, test_multiclass_data):
        sal_a = test_data.loc[test_data['discipline'] == 'A']['salary']
        sal_b = test_data.loc[test_data['discipline'] == 'B']['salary']

        with pytest.raises(ValueError):
            tTest(y1=sal_a, paired=True)

        with pytest.raises(ValueError):
            tTest(y1=sal_a, y2=sal_b, paired=True)

        with pytest.raises(ValueError):
            tTest(sal_a, sal_b, alternative='asdh')

        with pytest.raises(ValueError):
            tTest(group=test_multiclass_data['spray'], y1=test_multiclass_data['count'])


@pytest.fixture
def sample1():
    x = 682
    n = 925

    return x, n


class TestBinomial(object):

    def test_binomial_twosided(self):
        x, n = sample1()
        binomial_test = BinomialTest(n=n, x=x)

        assert binomial_test.alternative == 'two-sided'
        np.testing.assert_almost_equal(binomial_test.p_value, 2.4913404672588513e-13)

        agresti_coull_interval = {'conf level': 0.95,
                                  'interval': (0.7079790581519885, 0.7646527304391209),
                                  'probability of success': 0.7363158942955547}

        arcsine_interval = {'conf level': 0.95,
                            'interval': (0.708462749220724, 0.7651467076803447),
                            'probability of success': 0.7372972972972973,
                            'probability variance': 0.00020939458669772768}

        clopper_pearson_interval = {'conf level': 0.95,
                                    'interval': (0.7076682640790369, 0.7654065582415227),
                                    'probability of success': 0.7372972972972973}

        wilson_score_interval = {'conf level': 0.95,
                                 'interval': (0.46782780413153596, 0.5321721958684641),
                                 'probability of success': 0.5}

        # test_summary = {'Number of Successes': 682,
        #                 'Number of Trials': 925,
        #                 'alpha': 0.05,
        #                 'intervals': {'Agresti-Coull': {'conf level': 0.95,
        #                                                 'interval': (0.7079790581519885, 0.7646527304391209),
        #                                                 'probability of success': 0.7363158942955547},
        #                               'Arcsine Transform': {'conf level': 0.95,
        #                                                     'interval': (0.708462749220724, 0.7651467076803447),
        #                                                     'probability of success': 0.7372972972972973,
        #                                                     'probability variance': 0.00020939458669772768},
        #                               'Clopper-Pearson': {'conf level': 0.95,
        #                                                   'interval': (0.7076682640790369, 0.7654065582415227),
        #                                                   'probability of success': 0.7372972972972973},
        #                               'Wilson Score': {'conf level': 0.95,
        #                                                'interval': (0.46782780413153596, 0.5321721958684641),
        #                                                'probability of success': 0.5}},
        #                 'p-value': 2.4913404672588513e-13}

        assert binomial_test.agresti_coull_interval == agresti_coull_interval

        np.testing.assert_almost_equal(binomial_test.arcsine_transform_interval['conf level'],
                                       arcsine_interval['conf level'])
        np.testing.assert_almost_equal(binomial_test.arcsine_transform_interval['probability of success'],
                                       arcsine_interval['probability of success'])
        np.testing.assert_almost_equal(binomial_test.arcsine_transform_interval['probability variance'],
                                       arcsine_interval['probability variance'])
        np.testing.assert_array_almost_equal(binomial_test.arcsine_transform_interval['interval'],
                                             arcsine_interval['interval'])

        assert binomial_test.clopper_pearson_interval == clopper_pearson_interval

        assert binomial_test.wilson_score_interval == wilson_score_interval

        #assert binomial_test.test_summary == test_summary

    def test_binomial_less(self):
        x, n = sample1()
        binomial_test = BinomialTest(n=n, x=x, alternative='less')

        assert binomial_test.alternative == 'less'
        np.testing.assert_almost_equal(binomial_test.p_value, 0.9999999999997509)

        agresti_coull_interval = {'conf level': 0.95,
                                  'interval': (0.0, 0.7646527304391209),
                                  'probability of success': 0.7363158942955547}

        arcsine_interval = {'conf level': 0.95,
                            'interval': (0.0, 0.7651467076803447),
                            'probability of success': 0.7372972972972973,
                            'probability variance': 0.00020939458669772768}

        clopper_pearson_interval = {'conf level': 0.95,
                                    'interval': (0.0, 0.7610552746895429),
                                    'probability of success': 0.7372972972972973}

        wilson_score_interval = {'conf level': 0.95,
                                 'interval': (0.0, 0.5321721958684641),
                                 'probability of success': 0.5}

        # test_summary = {'Number of Successes': 682,
        #                 'Number of Trials': 925,
        #                 'alpha': 0.05,
        #                 'intervals': {'Agresti-Coull': {'conf level': 0.95,
        #                                                 'interval': (0.0, 0.7646527304391209),
        #                                                 'probability of success': 0.7363158942955547},
        #                               'Arcsine Transform': {'conf level': 0.95,
        #                                                     'interval': (0.0, 0.7651467076803447),
        #                                                     'probability of success': 0.7372972972972973,
        #                                                     'probability variance': 0.00020939458669772768},
        #                               'Clopper-Pearson': {'conf level': 0.95,
        #                                                   'interval': (0.0, 0.7610552746895429),
        #                                                   'probability of success': 0.7372972972972973},
        #                               'Wilson Score': {'conf level': 0.95,
        #                                                'interval': (0.0, 0.5321721958684641),
        #                                                'probability of success': 0.5}},
        #                 'p-value': 0.9999999999997509}

        assert binomial_test.agresti_coull_interval == agresti_coull_interval

        np.testing.assert_almost_equal(binomial_test.arcsine_transform_interval['conf level'],
                                       arcsine_interval['conf level'])
        np.testing.assert_almost_equal(binomial_test.arcsine_transform_interval['probability of success'],
                                       arcsine_interval['probability of success'])
        np.testing.assert_almost_equal(binomial_test.arcsine_transform_interval['probability variance'],
                                       arcsine_interval['probability variance'])
        np.testing.assert_array_almost_equal(binomial_test.arcsine_transform_interval['interval'],
                                             arcsine_interval['interval'])

        assert binomial_test.clopper_pearson_interval == clopper_pearson_interval

        assert binomial_test.wilson_score_interval == wilson_score_interval

        # assert binomial_test.test_summary == test_summary

    def test_binomial_greater(self):
        x, n = sample1()
        binomial_test = BinomialTest(n=n, x=x, alternative='greater')

        assert binomial_test.alternative == 'greater'

        np.testing.assert_almost_equal(binomial_test.p_value, 1.2569330927920093e-49)

        agresti_coull_interval = {'conf level': 0.95,
                                  'interval': (0.7603924379535446, 1.0),
                                  'probability of success': 0.7366052478060474}

        arcsine_interval = {'conf level': 0.95,
                            'interval': (0.7607405535791933, 1.0),
                            'probability of success': 0.7372972972972973,
                            'probability variance': 0.00020939458669772768}

        clopper_pearson_interval = {'conf level': 0.95,
                                    'interval': (0.7124129244365457, 1.0),
                                    'probability of success': 0.7372972972972973}

        wilson_score_interval = {'conf level': 0.95,
                                 'interval': (0.5270163003322376, 1.0),
                                 'probability of success': 0.5}

        # test_summary = {'Number of Successes': 682,
        #                 'Number of Trials': 925,
        #                 'alpha': 0.05,
        #                 'intervals': {'Agresti-Coull': {'conf level': 0.95,
        #                                                 'interval': (0.7603924379535446, 1.0),
        #                                                 'probability of success': 0.7366052478060474},
        #                               'Arcsine Transform': {'conf level': 0.95,
        #                                                     'interval': (0.7607405535791933, 1.0),
        #                                                     'probability of success': 0.7372972972972973,
        #                                                     'probability variance': 0.00020939458669772768},
        #                               'Clopper-Pearson': {'conf level': 0.95,
        #                                                   'interval': (0.7124129244365457, 1.0),
        #                                                   'probability of success': 0.7372972972972973},
        #                               'Wilson Score': {'conf level': 0.95,
        #                                                'interval': (0.5270163003322376, 1.0),
        #                                                'probability of success': 0.5}},
        #                 'p-value': 1.2569330927920093e-49}

        assert binomial_test.agresti_coull_interval == agresti_coull_interval

        np.testing.assert_almost_equal(binomial_test.arcsine_transform_interval['conf level'],
                                       arcsine_interval['conf level'])
        np.testing.assert_almost_equal(binomial_test.arcsine_transform_interval['probability of success'],
                                       arcsine_interval['probability of success'])
        np.testing.assert_almost_equal(binomial_test.arcsine_transform_interval['probability variance'],
                                       arcsine_interval['probability variance'])
        np.testing.assert_array_almost_equal(binomial_test.arcsine_transform_interval['interval'],
                                             arcsine_interval['interval'])

        assert binomial_test.clopper_pearson_interval == clopper_pearson_interval

        assert binomial_test.wilson_score_interval == wilson_score_interval

        # assert binomial_test.test_summary == test_summary

    def test_binomial_exceptions(self):
        x, n = sample1()

        with pytest.raises(ValueError):
            BinomialTest(x=200, n=100)

        with pytest.raises(ValueError):
            BinomialTest(x=x, n=n, p=2)

        with pytest.raises(ValueError):
            BinomialTest(x=x, n=n, alternative='na')
