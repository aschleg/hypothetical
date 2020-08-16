import pytest
from hypothetical import critical as c
import numpy as np
from numpy.testing import *


class TestChiSquareCritical(object):
    dof, alpha = 10, 0.05

    def test_critical_values(self):
        critical_value = c.chi_square_critical_value(self.alpha, self.dof)
        critical_value2 = c.chi_square_critical_value(str(self.alpha), str(self.dof))
        critical_value3 = c.chi_square_critical_value(str(self.alpha), float(self.dof))

        assert critical_value == 18.307
        assert critical_value2 == 18.307
        assert critical_value3 == 18.307

    def test_exceptions(self):
        with pytest.raises(ValueError):
            c.chi_square_critical_value(31, 0.05)
        with pytest.raises(ValueError):
            c.chi_square_critical_value(5, 1)
        with pytest.raises(ValueError):
            c.chi_square_critical_value(0.05, 31)


class TestUCritical(object):
    alpha = 0.05
    n, m = 10, 11

    def test_critical_values(self):
        critical_value = c.u_critical_value(self.n, self.m, self.alpha)
        critical_value2 = c.u_critical_value(str(self.n), str(self.m), str(self.alpha))

        assert critical_value == 31
        assert critical_value2 == 31

    def test_exceptions(self):
        with pytest.raises(ValueError):
            c.u_critical_value(31, 10, 0.05)
        with pytest.raises(ValueError):
            c.u_critical_value(10, 31, 0.05)
        with pytest.raises(ValueError):
            c.u_critical_value(10, 10, 0.50)


class TestWCritical(object):
    n, alpha, alternative = 15, 0.05, 'two-tail'

    def test_critical_values(self):
        crit_val = c.w_critical_value(self.n, self.alpha, self.alternative)
        crit_val2 = c.w_critical_value(str(self.n), str(self.alpha), self.alternative)

        assert crit_val == 25
        assert crit_val2 == 25

    def test_exceptions(self):
        with pytest.raises(ValueError):
            c.w_critical_value(31, 0.05, 'two-tail')
        with pytest.raises(ValueError):
            c.w_critical_value(20, 0.02, 'two-tail')
        with pytest.raises(ValueError):
            c.w_critical_value(25, 0.05, 'three-tail')


class TestRCritical(object):
    n1, n2, n3, n4 = 4, 20, 7, 15

    def test_critical_values(self):
        r_crit1, r_rcrit2 = c.r_critical_value(self.n1, self.n2)
        r_crit3, r_rcrit4 = c.r_critical_value(self.n3, self.n4)

        assert_allclose([r_crit1, r_rcrit2], [4, np.nan])
        assert_allclose([r_crit3, r_rcrit4], [6, 15])

    def test_exceptions(self):
        with pytest.raises(ValueError):
            c.r_critical_value(10, 25)
        with pytest.raises(ValueError):
            c.r_critical_value(25, 15)
