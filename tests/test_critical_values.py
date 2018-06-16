import pytest
from hypothetical import critical as c
import numpy as np


def test_chi_square_critical_value():
    dof, alpha = 10, 0.05

    critical_value = c.chi_square_critical_value(alpha, dof)
    critical_value2 = c.chi_square_critical_value(str(alpha), str(dof))
    critical_value3 = c.chi_square_critical_value(str(alpha), float(dof))

    assert critical_value == 18.307
    assert critical_value2 == 18.307
    assert critical_value3 == 18.307

    with pytest.raises(ValueError):
        c.chi_square_critical_value(31, 0.05)
    with pytest.raises(ValueError):
        c.chi_square_critical_value(5, 1)
    with pytest.raises(ValueError):
        c.chi_square_critical_value(0.05, 31)


def test_u_critical_value():
    alpha = 0.05
    n, m = 10, 11

    critical_value = c.u_critical_value(n, m, alpha)
    critical_value2 = c.u_critical_value(str(n), str(m), str(alpha))

    assert critical_value == 31
    assert critical_value2 == 31

    with pytest.raises(ValueError):
        c.u_critical_value(31, 10, 0.05)
    with pytest.raises(ValueError):
        c.u_critical_value(10, 31, 0.05)
    with pytest.raises(ValueError):
        c.u_critical_value(10, 10, 0.50)
    with pytest.raises(KeyError):
        c.u_critical_value(10, 8, 0.05)


def test_w_critical_value():
    n, alpha, alternative = 15, 0.05, 'two-tail'

    crit_val = c.w_critical_value(n, alpha, alternative)
    crit_val2 = c.w_critical_value(str(n), str(alpha), alternative)

    assert crit_val == 25
    assert crit_val2 == 25

    with pytest.raises(ValueError):
        c.w_critical_value(31, 0.05, 'two-tail')
    with pytest.raises(ValueError):
        c.w_critical_value(20, 0.02, 'two-tail')
    with pytest.raises(ValueError):
        c.w_critical_value(25, 0.05, 'three-tail')
