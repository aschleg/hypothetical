import os
import pytest
from hypothetical import posthoc
import pandas as pd


@pytest.fixture
def test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(datapath, '../data/PlantGrowth.csv'))

    return data


def test_tukeytest():
    pass
