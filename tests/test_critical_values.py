import pytest
from hypothetical import critical as c
import numpy as np


def test_w_critical_table():
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
