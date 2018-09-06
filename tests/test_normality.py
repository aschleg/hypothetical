import numpy as np
import pytest
from scipy.stats import chisquare

from hypothetical.normality import ChiSquareTest


class TestChiSquare(object):
    obs, exp = [29, 19, 18, 25, 17, 10, 15, 11], [18, 18, 18, 18, 18, 18, 18, 18]

    def test_chisquaretest(self):
        chi_test = ChiSquareTest(self.obs, self.exp)
        sci_chi_test = chisquare(self.obs, self.exp)

        np.testing.assert_almost_equal(chi_test.chi_square, sci_chi_test.statistic)
        np.testing.assert_almost_equal(chi_test.p_value, sci_chi_test.pvalue)

        assert not chi_test.continuity_correction
        assert chi_test.degrees_of_freedom == len(self.obs) - 1

    def test_chisquaretest_arr(self):
        chi_test = ChiSquareTest(np.array(self.obs), np.array(self.exp))
        sci_chi_test = chisquare(self.obs, self.exp)

        np.testing.assert_almost_equal(chi_test.chi_square, sci_chi_test.statistic)
        np.testing.assert_almost_equal(chi_test.p_value, sci_chi_test.pvalue)

        assert not chi_test.continuity_correction
        assert chi_test.degrees_of_freedom == len(self.obs) - 1

    def test_chisquaretest_continuity(self):
        chi_test = ChiSquareTest(self.obs, self.exp, continuity=True)

        np.testing.assert_almost_equal(chi_test.chi_square, 14.333333333333334)
        np.testing.assert_almost_equal(chi_test.p_value, 0.045560535300404756)

        assert chi_test.continuity_correction

    def test_chisquare_no_exp(self):
        chi_test = ChiSquareTest(self.obs)
        sci_chi_test = chisquare(self.obs, self.exp)

        np.testing.assert_almost_equal(chi_test.chi_square, sci_chi_test.statistic)
        np.testing.assert_almost_equal(chi_test.p_value, sci_chi_test.pvalue)

    def test_chisquare_exceptions(self):
        with pytest.raises(ValueError):
            ChiSquareTest(self.obs, self.exp[:5])
