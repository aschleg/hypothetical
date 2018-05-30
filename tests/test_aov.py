import pytest
from hypothetical.aov import anova_one_way
import numpy as np
import pandas as pd
import os


@pytest.fixture
def test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    plants = pd.read_csv(os.path.join(datapath, '../data/PlantGrowth.csv'))

    return plants


def test_anova_one_way():
    plants = test_data()

    anov = anova_one_way(plants['weight'], group=plants['group'])

    result = anov.summary()

    np.testing.assert_almost_equal(result['F statistic'], 4.846087862380137)
    np.testing.assert_almost_equal(result['group DoF'], 2)
    np.testing.assert_almost_equal(result['group Mean Squares'], 1.8831700000000007)
    np.testing.assert_almost_equal(result['group Sum of Squares'], 3.7663400000000014)

    np.testing.assert_almost_equal(result['p-value'], 0.01590995832562292)
    np.testing.assert_almost_equal(result['residual DoF'], 27)
    np.testing.assert_almost_equal(result['residual Mean Squares'], 0.38859592592592596)
    np.testing.assert_almost_equal(result['residual Sum of Squares'], 10.492090000000001)
