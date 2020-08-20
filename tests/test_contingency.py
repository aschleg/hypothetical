

import pytest
import numpy as np
from numpy.testing import *
from hypothetical.contingency import ChiSquareContingency, CochranQ, McNemarTest, \
    table_margins, expected_frequencies


class TestChiSquareContingency(object):
    observed = np.array([[23, 40, 16, 2], [11, 75, 107, 14], [1, 31, 60, 10]])
    expected = np.array([[7.3, 30.3, 38.0, 5.4], [18.6, 77.5, 97.1, 13.8], [9.1, 38.2, 47.9, 6.8]])

    def test_chi_square_contingency(self):
        c = ChiSquareContingency(self.observed, self.expected)

        assert_almost_equal(c.chi_square, 69.07632536255964)
        assert_almost_equal(c.p_value, 6.323684774702373e-13)

        assert_almost_equal(c.contingency_coefficient, 0.38790213046235816)
        assert_almost_equal(c.cramers_v, 0.2975893000268341)
        assert_almost_equal(np.absolute(c.phi_coefficient),
                            np.absolute(-0.4208548241150648))
        assert_almost_equal(c.tschuprows_coefficient, 0.25873655997832995)

        assert c.continuity
        assert c.degrees_freedom == 6

        observed2 = [[23, 40, 16, 2], [11, 75, 107, 14], [1, 31, 60, 10]]
        expected2 = [[7.3, 30.3, 38.0, 5.4], [18.6, 77.5, 97.1, 13.8], [9.1, 38.2, 47.9, 6.8]]

        c2 = ChiSquareContingency(observed2, expected2)

        assert_almost_equal(c.chi_square, c2.chi_square)

    def test_chi_square_contingency_no_continuity(self):
        obs = np.array([[23, 40], [11, 75]])
        exp = np.array([[7.3, 30.3], [18.6, 77.5]])

        c = ChiSquareContingency(obs, exp, continuity=False)

        assert_almost_equal(c.chi_square, 40.05705545808668)
        assert_almost_equal(c.p_value, 2.466522494156712e-10)
        assert_almost_equal(c.degrees_freedom, 1)

        assert_almost_equal(c.contingency_coefficient, 0.4603022164252613)
        assert_almost_equal(c.cramers_v, 0.5184971536820822)
        assert_almost_equal(c.phi_coefficient, 0.5184971536820822)

        assert not c.continuity

    def test_chi_square_contingency_no_expected(self):
        c = ChiSquareContingency(self.observed)

        assert_almost_equal(c.chi_square, 69.3893282675805)
        assert_almost_equal(c.p_value, 5.455268702303084e-13)
        assert_array_almost_equal(c.expected, np.array([[7.26923077, 30.32307692, 38.00769231,  5.4],
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

        assert_almost_equal(c.q_statistic, 16.666666666666668)
        assert_almost_equal(c.p_value, 0.00024036947641951404)
        assert c.degrees_freedom == 2

    def test_cochranq_exceptions(self):
        r1, r2, r3 = [0, 1, 1], [1, 1, 0], [0, 0, 1, 1]

        with pytest.raises(ValueError):
            CochranQ(r1, r2, r3)


class TestMcNemarTest(object):
    sample_data = np.array([[59, 6], [16, 80]])

    # def test_mcnemartest(self):
    #     m = McNemarTest(self.sample_data)
    #
    #     assert_almost_equal(m.exact_p_value, 0.052478790283203125)
    #     assert_almost_equal(m.mid_p_value, 0.034689664840698256)
    #     assert_almost_equal(m.mcnemar_p_value, 0.08981434395829568)
    #     assert_almost_equal(m.z_asymptotic_statistic, 2.1320071635561044)
    #     assert_almost_equal(m.mcnemar_x2_statistic, 4.545454545454546)
    #
    #     assert m.n == np.sum(self.sample_data)
    #     assert not m.continuity
    #
    #     sample_data_list = [[59, 6], [16, 80]]
    #
    #     m2 = McNemarTest(sample_data_list)
    #
    #     assert_almost_equal(m.exact_p_value, m2.exact_p_value)
    #     assert_almost_equal(m.mid_p_value, m2.mid_p_value)
    #     assert_almost_equal(m.mcnemar_p_value, m2.mcnemar_p_value)
    #     assert_almost_equal(m.z_asymptotic_statistic, m2.z_asymptotic_statistic)
    #     assert_almost_equal(m.mcnemar_x2_statistic, m2.mcnemar_x2_statistic)

    def test_mcnemartest_exceptions(self):

        with pytest.raises(ValueError):
            McNemarTest(np.array([[59, 6], [16, 80], [101, 100]]))

        with pytest.raises(ValueError):
            McNemarTest(np.array([[-1, 10], [10, 20]]))

        with pytest.raises(ValueError):
            McNemarTest(np.array([[0, 0], [0, 0]]))

        with pytest.raises(ValueError):
            McNemarTest(np.zeros((2, 2, 2)))


class TestTableMargins(object):
    cont_table = [[10, 10, 20], [20, 20, 10]]
    cont_table2 = np.array([[[10, 10, 20], [20, 20, 10]]])
    cont_table3 = np.array([10, 10, 20])

    def test_table_margins(self):
        t = table_margins(self.cont_table)
        t2 = table_margins(self.cont_table3)

        assert t[0][0][0] == 40
        assert all(t2[0] == self.cont_table3)

    def test_margins_exceptions(self):
        with pytest.raises(ValueError):
            table_margins(self.cont_table2)

    def test_expected_frequencies(self):
        e = expected_frequencies(self.cont_table)

        assert_array_almost_equal(e, np.array([[13.33333333, 13.33333333, 13.33333333],
                                                          [16.66666667, 16.66666667, 16.66666667]]))

    def test_expected_frequencies_exceptions(self):
        with pytest.raises(ValueError):
            expected_frequencies(self.cont_table2)
