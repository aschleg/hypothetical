import pandas as pd


class ChiSquare(object):

    def __init__(self, observed, expected, group=None):

        if len(observed) != len(expected):
            raise ValueError('number of observations must be of the same length as expected values.')

        self.observed = observed
        self.expected = expected
        self.group = group

    def _chisquare_value(self):
        pass

    def _p_value(self):
        pass
