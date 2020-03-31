

import pytest
from hypothetical.hypothesis import BinomialTest, tTest
import pandas as pd
import numpy as np
from numpy.testing import *
import os
from scipy.stats import t


def test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    salaries = pd.read_csv(os.path.join(datapath, 'data/Salaries.csv'))

    return salaries


def test_multiclass_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    insectsprays = pd.read_csv(os.path.join(datapath, 'data/InsectSprays.csv'))

    return insectsprays


class TestBinomial(object):
    x = 682
    n = 925

    def test_binomial_twosided(self):
        binomial_test = BinomialTest(n=self.n, x=self.x)

        assert binomial_test.alternative == 'two-sided'
        assert_almost_equal(binomial_test.p_value, 2.4913404672588513e-13)

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

        assert binomial_test.agresti_coull_interval == agresti_coull_interval

        assert_almost_equal(binomial_test.arcsine_transform_interval['conf level'],
                                       arcsine_interval['conf level'])
        assert_almost_equal(binomial_test.arcsine_transform_interval['probability of success'],
                                       arcsine_interval['probability of success'])
        assert_almost_equal(binomial_test.arcsine_transform_interval['probability variance'],
                                       arcsine_interval['probability variance'])
        assert_array_almost_equal(binomial_test.arcsine_transform_interval['interval'],
                                             arcsine_interval['interval'])

        assert binomial_test.clopper_pearson_interval == clopper_pearson_interval

        assert binomial_test.wilson_score_interval == wilson_score_interval

    def test_binomial_less(self):
        binomial_test = BinomialTest(n=self.n, x=self.x, alternative='less')

        assert binomial_test.alternative == 'less'
        assert_almost_equal(binomial_test.p_value, 0.9999999999997509)

        agresti_coull_interval = {'conf level': 0.95,
                                  'interval': (0.0, 0.7603924379535446),
                                  'probability of success': 0.7366052478060474}

        assert_array_almost_equal(binomial_test.agresti_coull_interval['interval'],
                                             agresti_coull_interval['interval'])

        assert_almost_equal(binomial_test.agresti_coull_interval['probability of success'],
                                       agresti_coull_interval['probability of success'])

        arcsine_interval = {'conf level': 0.95,
                            'interval': (0.0, 0.7607405535791933),
                            'probability of success': 0.7372972972972973,
                            'probability variance': 0.00020939458669772768}

        assert_almost_equal(binomial_test.arcsine_transform_interval['conf level'],
                                       arcsine_interval['conf level'])
        assert_almost_equal(binomial_test.arcsine_transform_interval['probability of success'],
                                       arcsine_interval['probability of success'])
        assert_almost_equal(binomial_test.arcsine_transform_interval['probability variance'],
                                       arcsine_interval['probability variance'])
        assert_array_almost_equal(binomial_test.arcsine_transform_interval['interval'],
                                             arcsine_interval['interval'])

        clopper_pearson_interval = {'conf level': 0.95,
                                    'interval': (0.0, 0.7610552746895429),
                                    'probability of success': 0.7372972972972973}

        wilson_score_interval = {'conf level': 0.95,
                                 'interval': (0.0, 0.5270163003322376),
                                 'probability of success': 0.5}

        assert binomial_test.clopper_pearson_interval == clopper_pearson_interval

        assert binomial_test.wilson_score_interval == wilson_score_interval

    def test_binomial_greater(self):
        binomial_test = BinomialTest(n=self.n, x=self.x, alternative='greater')

        assert binomial_test.alternative == 'greater'

        assert_almost_equal(binomial_test.p_value, 1.2569330927920093e-49)

        agresti_coull_interval = {'conf level': 0.95,
                                  'interval': (0.7603924379535446, 1.0),
                                  'probability of success': 0.7366052478060474}

        assert_array_almost_equal(binomial_test.agresti_coull_interval['interval'],
                                             agresti_coull_interval['interval'])

        assert_almost_equal(binomial_test.agresti_coull_interval['probability of success'],
                                       agresti_coull_interval['probability of success'])

        arcsine_interval = {'conf level': 0.95,
                            'interval': (0.7607405535791933, 1.0),
                            'probability of success': 0.7372972972972973,
                            'probability variance': 0.00020939458669772768}

        assert_almost_equal(binomial_test.arcsine_transform_interval['conf level'],
                                       arcsine_interval['conf level'])
        assert_almost_equal(binomial_test.arcsine_transform_interval['probability of success'],
                                       arcsine_interval['probability of success'])
        assert_almost_equal(binomial_test.arcsine_transform_interval['probability variance'],
                                       arcsine_interval['probability variance'])
        assert_array_almost_equal(binomial_test.arcsine_transform_interval['interval'],
                                             arcsine_interval['interval'])

        clopper_pearson_interval = {'conf level': 0.95,
                                    'interval': (0.7124129244365457, 1.0),
                                    'probability of success': 0.7372972972972973}

        wilson_score_interval = {'conf level': 0.95,
                                 'interval': (0.5270163003322376, 1.0),
                                 'probability of success': 0.5}

        assert binomial_test.agresti_coull_interval == agresti_coull_interval

        assert binomial_test.clopper_pearson_interval == clopper_pearson_interval

        assert binomial_test.wilson_score_interval == wilson_score_interval

    def test_binomial_no_continuity(self):
        binomial_test = BinomialTest(x=self.x, n=self.n, p=0.7, continuity=False)

        wilson_score_interval = {'conf level': 0.95,
                                 'interval': (0.66969078194969, 0.7286549168628622),
                                 'probability of success': 0.6991728494062761}

        assert_array_almost_equal(binomial_test.wilson_score_interval['interval'],
                                             wilson_score_interval['interval'])

        assert_almost_equal(binomial_test.wilson_score_interval['probability of success'],
                                       wilson_score_interval['probability of success'])

    def test_binomial_no_continuity_greater(self):
        binomial_test = BinomialTest(x=self.x, n=self.n, p=0.7, continuity=False, alternative='greater')

        wilson_score_interval = {'conf level': 0.95,
                                 'interval': (0.7241711245792711, 1.0),
                                 'probability of success': 0.6994167236634111}

        assert_array_almost_equal(binomial_test.wilson_score_interval['interval'],
                                             wilson_score_interval['interval'])

        assert_almost_equal(binomial_test.wilson_score_interval['probability of success'],
                                       wilson_score_interval['probability of success'])

    def test_binomial_no_continuity_less(self):
        binomial_test = BinomialTest(x=self.x, n=self.n, p=0.7, continuity=False, alternative='less')

        wilson_score_interval = {'conf level': 0.95,
                                 'interval': (0.0, 0.724171124579271),
                                 'probability of success': 0.6994167236634111}

        assert_array_almost_equal(binomial_test.wilson_score_interval['interval'],
                                             wilson_score_interval['interval'])

        assert_almost_equal(binomial_test.wilson_score_interval['probability of success'],
                                       wilson_score_interval['probability of success'])

    def test_binomial_exceptions(self):

        with pytest.raises(ValueError):
            BinomialTest(x=200, n=100)

        with pytest.raises(ValueError):
            BinomialTest(x=self.x, n=self.n, p=2)

        with pytest.raises(ValueError):
            BinomialTest(x=self.x, n=self.n, alternative='na')


class Test_tTest(object):

    data = test_data()
    multiclass_data = test_multiclass_data()

    def test_two_sample_welch_test(self):

        sal_a = self.data.loc[self.data['discipline'] == 'A']['salary']
        sal_b = self.data.loc[self.data['discipline'] == 'B']['salary']

        ttest = tTest(y1=sal_a, y2=sal_b)

        test_summary = ttest.test_summary

        assert_almost_equal(test_summary['Sample 1 Mean'], np.mean(sal_a))
        assert_almost_equal(test_summary['Sample 2 Mean'], np.mean(sal_b))
        assert_almost_equal(test_summary['t-statistic'], -3.1386989278486013)
        assert_almost_equal(test_summary['degrees of freedom'], 377.89897288941387)
        assert_almost_equal(test_summary['p-value'], t.cdf(test_summary['t-statistic'],
                                                                      test_summary['degrees of freedom']) * 2)

        assert test_summary['alternative'] == 'two-sided'
        assert test_summary['test description'] == "Two-Sample Welch's t-test"

        ttest_group = tTest(group=self.data['discipline'], y1=self.data['salary'])
        test_group_summary = ttest_group.test_summary

        assert_almost_equal(test_summary['Sample 1 Mean'], test_group_summary['Sample 1 Mean'])
        assert_almost_equal(test_summary['Sample 2 Mean'], test_group_summary['Sample 2 Mean'])
        assert_almost_equal(test_summary['p-value'], test_group_summary['p-value'])
        assert_almost_equal(test_summary['degrees of freedom'], test_group_summary['degrees of freedom'], 5)
        assert_almost_equal(test_summary['t-statistic'], test_group_summary['t-statistic'])

        assert test_group_summary['alternative'] == 'two-sided'
        assert test_group_summary['test description'] == "Two-Sample Welch's t-test"

    def test_two_sample_students_test(self):

        sal_a = self.data.loc[self.data['discipline'] == 'A']['salary']
        sal_b = self.data.loc[self.data['discipline'] == 'B']['salary']

        ttest = tTest(y1=sal_a, y2=sal_b, var_equal=True)

        test_summary = ttest.test_summary

        assert_almost_equal(test_summary['Sample 1 Mean'], np.mean(sal_a))
        assert_almost_equal(test_summary['Sample 2 Mean'], np.mean(sal_b))
        assert_almost_equal(test_summary['t-statistic'], -3.1485647713976195)
        assert_almost_equal(test_summary['p-value'], t.cdf(test_summary['t-statistic'],
                                                                      test_summary['degrees of freedom']) * 2)

        assert test_summary['alternative'] == 'two-sided'
        assert test_summary['test description'] == "Two-Sample Student's t-test"

        assert len(sal_a) + len(sal_b) - 2 == test_summary['degrees of freedom']

    def test_one_sample_test(self):
        sal_a = self.data.loc[self.data['discipline'] == 'A']['salary']

        ttest = tTest(y1=sal_a)

        test_summary = ttest.test_summary

        assert_almost_equal(test_summary['Sample 1 Mean'], np.mean(sal_a))
        assert_almost_equal(test_summary['t-statistic'], 47.95382017797468)
        assert_almost_equal(test_summary['p-value'], 2.220446049250313e-16)

        assert test_summary['alternative'] == 'two-sided'
        assert test_summary['test description'] == 'One-Sample t-test'

        assert len(sal_a) - 1 == test_summary['degrees of freedom']

        ttest_mu = tTest(y1=sal_a, mu=100000)

        test_mu_summary = ttest_mu._generate_result_summary()

        assert_almost_equal(test_mu_summary['Sample 1 Mean'], np.mean(sal_a))
        assert_almost_equal(test_mu_summary['p-value'], 0.0002159346891279501)
        assert_almost_equal(test_mu_summary['t-statistic'], 3.776470249422699)

        assert test_mu_summary['alternative'] == 'two-sided'
        assert test_mu_summary['test description'] == 'One-Sample t-test'
        assert test_mu_summary['mu'] == 100000
        assert len(sal_a) - 1 == test_mu_summary['degrees of freedom']

    def test_paired_sample_test(self):
        sal_a = self.data.loc[self.data['discipline'] == 'A']['salary']
        sal_b = self.data.loc[self.data['discipline'] == 'B']['salary']
        sal_b2 = sal_b[0:len(sal_a)]

        ttest = tTest(y1=sal_a, y2=sal_b2, paired=True)

        test_summary = ttest.test_summary

        assert_almost_equal(test_summary['Sample Difference Mean'], np.mean(np.array(sal_a) - np.array(sal_b2)))
        assert_almost_equal(test_summary['t-statistic'], -2.3158121700626406)
        assert_almost_equal(test_summary['p-value'], t.cdf(test_summary['t-statistic'],
                                                                      test_summary['degrees of freedom']) * 2)

        assert test_summary['alternative'] == 'two-sided'
        assert test_summary['test description'] == 'Paired t-test'

        assert len(sal_a) - 1 == test_summary['degrees of freedom']

    def test_alternatives(self):
        sal_a = self.data.loc[self.data['discipline'] == 'A']['salary']
        sal_b = self.data.loc[self.data['discipline'] == 'B']['salary']

        ttest = tTest(y1=sal_a, y2=sal_b, alternative='greater')

        test_summary = ttest.test_summary

        assert_almost_equal(test_summary['p-value'], 0.9990848459959981)
        assert_almost_equal(test_summary['t-statistic'], -3.1386989278486013)

        assert test_summary['alternative'] == 'greater'

        ttest_less = tTest(y1=sal_a, y2=sal_b, alternative='less')

        test_less_summary = ttest_less.test_summary

        assert test_less_summary['alternative'] == 'less'
        assert_almost_equal(test_less_summary['t-statistic'], -3.1386989278486013)
        assert_almost_equal(test_less_summary['p-value'], 0.0009151540040019292)

    def test_ttest_exceptions(self):
        sal_a = self.data.loc[self.data['discipline'] == 'A']['salary']
        sal_b = self.data.loc[self.data['discipline'] == 'B']['salary']

        with pytest.raises(ValueError):
            tTest(y1=sal_a, paired=True)

        with pytest.raises(ValueError):
            tTest(y1=sal_a, y2=sal_b, paired=True)

        with pytest.raises(ValueError):
            tTest(sal_a, sal_b, alternative='asdh')

        with pytest.raises(ValueError):
            tTest(group=self.multiclass_data['spray'], y1=self.multiclass_data['count'])
