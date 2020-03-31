import pytest
import numpy as np
from numpy.testing import *
import pandas as pd
from hypothetical.descriptive import covar, pearson, spearman, var, std_dev, variance_condition, \
    kurtosis, skewness, mean_absolute_deviation
from scipy.stats import spearmanr
from numpy.core.multiarray import array


class TestCorrelationCovariance(object):

    d = np.array([[ 1.   ,  1.11 ,  2.569,  3.58 ,  0.76 ],
       [ 1.   ,  1.19 ,  2.928,  3.75 ,  0.821],
       [ 1.   ,  1.09 ,  2.865,  3.93 ,  0.928],
       [ 1.   ,  1.25 ,  3.844,  3.94 ,  1.009],
       [ 1.   ,  1.11 ,  3.027,  3.6  ,  0.766],
       [ 1.   ,  1.08 ,  2.336,  3.51 ,  0.726],
       [ 1.   ,  1.11 ,  3.211,  3.98 ,  1.209],
       [ 1.   ,  1.16 ,  3.037,  3.62 ,  0.75 ],
       [ 2.   ,  1.05 ,  2.074,  4.09 ,  1.036],
       [ 2.   ,  1.17 ,  2.885,  4.06 ,  1.094],
       [ 2.   ,  1.11 ,  3.378,  4.87 ,  1.635],
       [ 2.   ,  1.25 ,  3.906,  4.98 ,  1.517],
       [ 2.   ,  1.17 ,  2.782,  4.38 ,  1.197],
       [ 2.   ,  1.15 ,  3.018,  4.65 ,  1.244],
       [ 2.   ,  1.17 ,  3.383,  4.69 ,  1.495],
       [ 2.   ,  1.19 ,  3.447,  4.4  ,  1.026],
       [ 3.   ,  1.07 ,  2.505,  3.76 ,  0.912],
       [ 3.   ,  0.99 ,  2.315,  4.44 ,  1.398],
       [ 3.   ,  1.06 ,  2.667,  4.38 ,  1.197],
       [ 3.   ,  1.02 ,  2.39 ,  4.67 ,  1.613],
       [ 3.   ,  1.15 ,  3.021,  4.48 ,  1.476],
       [ 3.   ,  1.2  ,  3.085,  4.78 ,  1.571],
       [ 3.   ,  1.2  ,  3.308,  4.57 ,  1.506],
       [ 3.   ,  1.17 ,  3.231,  4.56 ,  1.458],
       [ 4.   ,  1.22 ,  2.838,  3.89 ,  0.944],
       [ 4.   ,  1.03 ,  2.351,  4.05 ,  1.241],
       [ 4.   ,  1.14 ,  3.001,  4.05 ,  1.023],
       [ 4.   ,  1.01 ,  2.439,  3.92 ,  1.067],
       [ 4.   ,  0.99 ,  2.199,  3.27 ,  0.693],
       [ 4.   ,  1.11 ,  3.318,  3.95 ,  1.085],
       [ 4.   ,  1.2  ,  3.601,  4.27 ,  1.242],
       [ 4.   ,  1.08 ,  3.291,  3.85 ,  1.017],
       [ 5.   ,  0.91 ,  1.532,  4.04 ,  1.084],
       [ 5.   ,  1.15 ,  2.552,  4.16 ,  1.151],
       [ 5.   ,  1.14 ,  3.083,  4.79 ,  1.381],
       [ 5.   ,  1.05 ,  2.33 ,  4.42 ,  1.242],
       [ 5.   ,  0.99 ,  2.079,  3.47 ,  0.673],
       [ 5.   ,  1.22 ,  3.366,  4.41 ,  1.137],
       [ 5.   ,  1.05 ,  2.416,  4.64 ,  1.455],
       [ 5.   ,  1.13 ,  3.1  ,  4.57 ,  1.325],
       [ 6.   ,  1.11 ,  2.813,  3.76 ,  0.8  ],
       [ 6.   ,  0.75 ,  0.84 ,  3.14 ,  0.606],
       [ 6.   ,  1.05 ,  2.199,  3.75 ,  0.79 ],
       [ 6.   ,  1.02 ,  2.132,  3.99 ,  0.853],
       [ 6.   ,  1.05 ,  1.949,  3.34 ,  0.61 ],
       [ 6.   ,  1.07 ,  2.251,  3.21 ,  0.562],
       [ 6.   ,  1.13 ,  3.064,  3.63 ,  0.707],
       [ 6.   ,  1.11 ,  2.469,  3.95 ,  0.952]])

    def test_naive_covariance(self):
        assert_allclose(covar(self.d[:, 1:], method='naive'),
                                   np.cov(self.d[:, 1:], rowvar=False))

        assert_allclose(covar(self.d[:, 1:3], self.d[:, 3:], 'naive'),
                                   np.cov(self.d[:, 1:], rowvar=False))

    def test_shifted_covariance(self):
        assert_allclose(covar(self.d[:, 1:], method='shifted covariance'),
                                   np.cov(self.d[:, 1:], rowvar=False))

        assert_allclose(covar(self.d[:, 1:3], self.d[:, 3:], 'shifted covariance'),
                                   np.cov(self.d[:, 1:], rowvar=False))

    def test_two_pass_covariance(self):
        assert_allclose(covar(self.d[:, 1:], method='two-pass covariance'),
                                   np.cov(self.d[:, 1:], rowvar=False))

        assert_allclose(covar(self.d[:, 1:3], self.d[:, 3:], 'two-pass covariance'),
                                   np.cov(self.d[:, 1:], rowvar=False))

    def test_covar_no_method(self):
        with pytest.raises(ValueError):
            covar(self.d[:, 1:3], self.d[:, 3:], 'NA_METHOD')

    def test_pearson(self):
        assert_allclose(pearson(self.d[:, 1:]),
                                   np.corrcoef(self.d[:, 1:], rowvar=False))

        assert_allclose(pearson(self.d[:, 1:3], self.d[:, 3:]),
                                   np.corrcoef(self.d[:, 1:], rowvar=False))

    def test_spearman(self):
        assert_allclose(spearman(self.d[:, 1:]),
                                   spearmanr(self.d[:, 1:])[0])

        assert_allclose(spearman(self.d[:, 1:3], self.d[:, 3:]),
                                   spearmanr(self.d[:, 1:])[0])


class TestVariance(object):

    f = pd.DataFrame({0: [1, -1, 2, 2], 1: [-1, 2, 1, -1], 2: [2, 1, 3, 2], 3: [2, -1, 2, 1]})
    h = [[16, 4, 8, 4], [4, 10, 8, 4], [8, 8, 12, 10], [4, 4, 10, 12]]
    fa = np.array(f)

    def test_var_corrected_two_pass(self):
        assert_allclose(np.array(var(self.f)).reshape(4,), np.array([2, 2.25, 0.666667, 2]), rtol=1e-02)
        assert_allclose(np.array(var(self.f, 'corrected two pass')).reshape(4,),
                                   np.array([2, 2.25, 0.666667, 2]), rtol=1e-02)

        assert_allclose(var(self.h).reshape(4,), np.array([32, 9, 3.666667, 17]), rtol=1e-02)

    def test_var_textbook_one_pass(self):
        assert_allclose(np.array(var(self.f, 'textbook one pass')).reshape(4,),
                                   np.array([2, 2.25, 0.666667, 2]), rtol=1e-02)

        assert_allclose(np.array(var(self.h, 'textbook one pass')).reshape(4,),
                                   np.array([32, 9, 3.666667, 17]), rtol=1e-02)

        assert_almost_equal(var(self.fa[:, 2], 'textbook one pass'), 0.66666666666666663)

    def test_var_standard_two_pass(self):
        assert_allclose(np.array(var(self.f, 'standard two pass')).reshape(4,),
                                   np.array([2, 2.25, 0.666667, 2]), rtol=1e-02)

        assert_allclose(np.array(var(self.h, 'standard two pass')).reshape(4,),
                                   np.array([32, 9, 3.666667, 17]), rtol=1e-02)

        assert_equal(var(self.fa[:, 1], 'standard two pass'), 2.25)

    def test_var_youngs_cramer(self):
        assert_allclose(np.array(var(self.f, 'youngs cramer')).reshape(4,),
                                   np.array([2, 2.25, 0.666667, 2]), rtol=1e-02)

        assert_allclose(np.array(var(self.h, 'youngs cramer')).reshape(4,),
                                   np.array([32, 9, 3.666667, 17]), rtol=1e-02)

        assert_equal(var(self.fa[:, 1], 'youngs cramer'), 2.25)

    def test_stddev(self):
        assert_equal(std_dev(self.fa[:, 1]), 1.5)
        assert_allclose(std_dev(self.fa), array([ 1.41421356,  1.5       ,  0.81649658,  1.41421356]))

    def test_var_cond(self):
        assert_almost_equal(variance_condition(self.fa[:, 1]), 1.7638342073763937)
        assert_allclose(variance_condition(self.fa), array([2.23606798, 1.76383421, 5.19615242, 2.23606798]))

        assert_allclose(variance_condition(pd.DataFrame(self.fa)),
                                   array([2.23606798, 1.76383421, 5.19615242, 2.23606798]))

        assert_allclose(variance_condition(list(self.fa)),
                                   array([2.23606798, 1.76383421, 5.19615242, 2.23606798]))

        ff = np.array([np.array(self.f), np.array(self.f)])

        with pytest.raises(ValueError):
            variance_condition(ff)

    def test_errors(self):
        with pytest.raises(ValueError):
            var(self.f, 'NA')

        ff = np.array([np.array(self.f), np.array(self.f)])

        with pytest.raises(ValueError):
            var(ff)


class TestKurtosis(object):

    s1 = [5, 2, 4, 5, 6, 2, 3]
    s2 = [4, 6, 4, 3, 2, 6, 7]

    def test_exceptions(self):
        with pytest.raises(ValueError):
            kurtosis(self.s1, axis=2)
        with pytest.raises(ValueError):
            kurtosis(np.zeros((4, 4, 4)))

    def test_kurtosis(self):
        k1 = kurtosis(self.s1)
        k2 = kurtosis([self.s1, self.s2], axis=1)

        assert_almost_equal(k1, -1.4515532544378704)
        assert_allclose(k2, array([-1.45155325, -1.32230624]))


class TestSkewness(object):

    s1 = [5, 2, 4, 5, 6, 2, 3]
    s2 = [4, 6, 4, 3, 2, 6, 7]

    def test_exceptions(self):
        with pytest.raises(ValueError):
            skewness(self.s1, axis=2)
        with pytest.raises(ValueError):
            skewness(np.zeros((4, 4, 4)))

    def test_skewness(self):
        s1 = skewness(self.s1)
        s2 = skewness([self.s1, self.s2], axis=1)

        assert_almost_equal(s1, -0.028285981029545847)
        assert_allclose(s2, array([-0.02828598, -0.03331004]))


class TestMeanAbsoluteDeviation(object):

    s1 = [2, 2, 3, 4, 5, 5, 6]

    def test_exceptions(self):
        with pytest.raises(ValueError):
            mean_absolute_deviation(self.s1, axis=2)
        with pytest.raises(TypeError):
            mean_absolute_deviation(self.s1, mean='true')
        with pytest.raises(ValueError):
            mean_absolute_deviation(np.zeros((4, 4, 4)))

    def test_mean_deviation(self):
        pass
