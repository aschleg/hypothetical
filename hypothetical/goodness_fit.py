import numpy as np
import pandas as pd
from hypothetical._lib import build_des_mat


class ChiSquare(object):

    def __init__(self, observed, expected=None, group=None, continuity=False):

        if not isinstance(observed, np.ndarray):
            self.observed = np.array(observed)
        else:
            self.observed = observed

        if expected is None:
            obs_mean = np.mean(self.observed)
            self.expected = np.full_like(self.observed, obs_mean)

        else:
            if not isinstance(expected, np.ndarray):
                self.expected = np.array(expected)
            else:
                self.expected = expected

            if self.observed.shape[0] != self.expected.shape[0]:
                raise ValueError('number of observations must be of the same length as expected values.')

        self.group = group

        if self.group is not None:
            self._design_matrix = build_des_mat(self.observed, self.expected, group=self.group)
        else:
            self._design_matrix = build_des_mat(self.observed, self.expected)

        self.n = self.observed.shape[0]

        self.continuity_correction = continuity

    def _chisquare_value(self):
        pass

    def _p_value(self):
        pass
