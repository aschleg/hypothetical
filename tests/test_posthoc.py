import os
import pytest
from hypothetical import posthoc
from numpy.testing import *
import pandas as pd


class TestGamesHowell(object):

    datapath = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(datapath, 'data/InsectSprays.csv'))

    def test_games_howell(self):
        sprays = self.data
        spray_test = posthoc.GamesHowell(sprays['count'], group=sprays['spray'])

        res = spray_test.test_result

        res_group_vec_expected = ['A : B',
                                  'A : C',
                                  'A : D',
                                  'A : E',
                                  'A : F',
                                  'B : C',
                                  'B : D',
                                  'B : E',
                                  'B : F',
                                  'C : D',
                                  'C : E',
                                  'C : F',
                                  'D : E',
                                  'D : F',
                                  'E : F']

        res_group_mean_diff_expected = [0.8333333333333339,
                                        -12.416666666666666,
                                        -9.583333333333332,
                                        -11.0,
                                        2.166666666666668,
                                        -13.25,
                                        -10.416666666666668,
                                        -11.833333333333334,
                                        1.333333333333334,
                                        2.8333333333333335,
                                        1.4166666666666665,
                                        14.583333333333334,
                                        -1.416666666666667,
                                        11.75,
                                        13.166666666666668]

        res_group_t_value_expected = [0.45352438521873684,
                                      8.407339499090465,
                                      6.214359489959841,
                                      7.579791383017868,
                                      0.9619438590966518,
                                      9.753917079412913,
                                      7.289021195232187,
                                      8.89397045271091,
                                      0.6125900134023213,
                                      3.078215365445631,
                                      1.8680395871105286,
                                      7.748439687475623,
                                      1.6122487000489365,
                                      6.076375289078413,
                                      7.07111931253599]

        assert res['groups'].tolist() == res_group_vec_expected

        assert_allclose(res['mean_difference'], res_group_mean_diff_expected, rtol=1e-3)
        assert_allclose(res['t_value'], res_group_t_value_expected, rtol=1e-3)


def test_tukeytest():
    pass
