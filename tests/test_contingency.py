

import pytest
import numpy as np
from hypothetical.contingency import FisherTest, McNemarTest


class TestFisherTest(object):
    pass


@pytest.fixture
def sample_data():

    return np.array([[59, 6], [16, 80]])


class TestMcNemarTest(object):

    def test_mcnemartest_exceptions(self):

        with pytest.raises(ValueError):
            McNemarTest(np.array([[59, 6], [16, 80], [101, 100]]))

        with pytest.raises(ValueError):
            McNemarTest(np.array([[-1, 10], [10, 20]]))

        with pytest.raises(ValueError):
            McNemarTest(table=sample_data(), alternative='na')
