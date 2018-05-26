import pytest
from hypy import critical as c
import numpy as np


def test_w_critical_table():
    n, alpha, alternative = 15, 0.05, 'two-tail'

    crit_val = c.w_critical_value(n, alpha, alternative)

    assert crit_val == 25
