

import pytest
import numpy as np
from hypothetical.contingency import ChiSquareContingency, CochranQ, FisherTest, McNemarTest


@pytest.fixture
def sample_data():

    return np.array([[59, 6], [16, 80]])


class TestChiSquareContingency(object):
    observed = np.array([[23, 40, 16, 2], [11, 75, 107, 14], [1, 31, 60, 10]])
    expected = np.array([[7.3, 30.3, 38.0, 5.4], [18.6, 77.5, 97.1, 13.8], [9.1, 38.2, 47.9, 6.8]])

    def test_chi_square_contingency(self):
        c = ChiSquareContingency(self.observed, self.expected)

        np.testing.assert_almost_equal(c.chi_square, 69.07632536255964)
        np.testing.assert_almost_equal(c.p_value, 6.323684774702373e-13)

        np.testing.assert_almost_equal(c.association_measures['C'], 0.38790213046235816)
        np.testing.assert_almost_equal(c.association_measures['Cramers V'], 0.2975893000268341)
        np.testing.assert_almost_equal(c.association_measures['phi-coefficient'], 0.4208548241150648)

        assert c.continuity
        assert c.degrees_freedom == 6

    def test_chi_square_contingency_no_continuity(self):
        obs = np.array([[23, 40], [11, 75]])
        exp = np.array([[7.3, 30.3], [18.6, 77.5]])

        c = ChiSquareContingency(obs, exp, continuity=False)

        np.testing.assert_almost_equal(c.chi_square, 40.05705545808668)
        np.testing.assert_almost_equal(c.p_value, 2.466522494156712e-10)
        np.testing.assert_almost_equal(c.degrees_freedom, 1)

        np.testing.assert_almost_equal(c.association_measures['C'], 0.4603022164252613)
        np.testing.assert_almost_equal(c.association_measures['Cramers V'], 0.5184971536820822)
        np.testing.assert_almost_equal(c.association_measures['phi-coefficient'], -0.5184971536820822)

        assert not c.continuity

    def test_chi_square_contingency_no_expected(self):
        c = ChiSquareContingency(self.observed)

        np.testing.assert_almost_equal(c.chi_square, 69.3893282675805)
        np.testing.assert_almost_equal(c.p_value, 5.455268702303084e-13)
        np.testing.assert_array_almost_equal(c.expected, np.array([[7.26923077, 30.32307692, 38.00769231,  5.4],
                                                                   [18.57692308, 77.49230769, 97.13076923, 13.8],
                                                                   [9.15384615, 38.18461538, 47.86153846,  6.8]]))

        assert c.degrees_freedom == 6

    def test_chi_square_exceptions(self):
        with pytest.raises(ValueError):
            ChiSquareContingency(self.observed, self.expected[:1])


class TestCochranQ(object):
    r1 = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    r2 = [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    r3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]

    def test_cochranq(self):
        c = CochranQ(self.r1, self.r2, self.r3)

        np.testing.assert_almost_equal(c.q_statistic, 16.666666666666668)
        np.testing.assert_almost_equal(c.p_value, 0.00024036947641951404)
        np.testing.assert_array_almost_equal(np.array(c.sample_summary_table), np.array([[0., 13., 18.,  5.],
                                                                                         [1., 13., 18.,  5.],
                                                                                         [2.,  3., 18., 15.]]))

        assert c.degrees_freedom == 2
        assert c.k == 3


class TestMcNemarTest(object):

    def test_mcnemartest_exceptions(self):

        with pytest.raises(ValueError):
            McNemarTest(np.array([[59, 6], [16, 80], [101, 100]]))

        with pytest.raises(ValueError):
            McNemarTest(np.array([[-1, 10], [10, 20]]))

        with pytest.raises(ValueError):
            McNemarTest(table=sample_data(), alternative='na')
