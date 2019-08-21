import pytest
from hypothetical.aov import AnovaOneWay, ManovaOneWay
import numpy as np
import pandas as pd
import os


def test_data():
    datapath = os.path.dirname(os.path.abspath(__file__))
    plants = pd.read_csv(os.path.join(datapath, 'data/PlantGrowth.csv'))

    return plants


def multivariate_test_data():
    d = np.array([[1., 1.11, 2.569, 3.58, 0.76],
                  [1., 1.19, 2.928, 3.75, 0.821],
                  [1., 1.09, 2.865, 3.93, 0.928],
                  [1., 1.25, 3.844, 3.94, 1.009],
                  [1., 1.11, 3.027, 3.6, 0.766],
                  [1., 1.08, 2.336, 3.51, 0.726],
                  [1., 1.11, 3.211, 3.98, 1.209],
                  [1., 1.16, 3.037, 3.62, 0.75],
                  [2., 1.05, 2.074, 4.09, 1.036],
                  [2., 1.17, 2.885, 4.06, 1.094],
                  [2., 1.11, 3.378, 4.87, 1.635],
                  [2., 1.25, 3.906, 4.98, 1.517],
                  [2., 1.17, 2.782, 4.38, 1.197],
                  [2., 1.15, 3.018, 4.65, 1.244],
                  [2., 1.17, 3.383, 4.69, 1.495],
                  [2., 1.19, 3.447, 4.4, 1.026],
                  [3., 1.07, 2.505, 3.76, 0.912],
                  [3., 0.99, 2.315, 4.44, 1.398],
                  [3., 1.06, 2.667, 4.38, 1.197],
                  [3., 1.02, 2.39, 4.67, 1.613],
                  [3., 1.15, 3.021, 4.48, 1.476],
                  [3., 1.2, 3.085, 4.78, 1.571],
                  [3., 1.2, 3.308, 4.57, 1.506],
                  [3., 1.17, 3.231, 4.56, 1.458],
                  [4., 1.22, 2.838, 3.89, 0.944],
                  [4., 1.03, 2.351, 4.05, 1.241],
                  [4., 1.14, 3.001, 4.05, 1.023],
                  [4., 1.01, 2.439, 3.92, 1.067],
                  [4., 0.99, 2.199, 3.27, 0.693],
                  [4., 1.11, 3.318, 3.95, 1.085],
                  [4., 1.2, 3.601, 4.27, 1.242],
                  [4., 1.08, 3.291, 3.85, 1.017],
                  [5., 0.91, 1.532, 4.04, 1.084],
                  [5., 1.15, 2.552, 4.16, 1.151],
                  [5., 1.14, 3.083, 4.79, 1.381],
                  [5., 1.05, 2.33, 4.42, 1.242],
                  [5., 0.99, 2.079, 3.47, 0.673],
                  [5., 1.22, 3.366, 4.41, 1.137],
                  [5., 1.05, 2.416, 4.64, 1.455],
                  [5., 1.13, 3.1, 4.57, 1.325],
                  [6., 1.11, 2.813, 3.76, 0.8],
                  [6., 0.75, 0.84, 3.14, 0.606],
                  [6., 1.05, 2.199, 3.75, 0.79],
                  [6., 1.02, 2.132, 3.99, 0.853],
                  [6., 1.05, 1.949, 3.34, 0.61],
                  [6., 1.07, 2.251, 3.21, 0.562],
                  [6., 1.13, 3.064, 3.63, 0.707],
                  [6., 1.11, 2.469, 3.95, 0.952]])

    return d


def test_AnovaOneWay():
    plants = test_data()

    anov = AnovaOneWay(plants['weight'], group=plants['group'])

    result = anov.test_summary

    np.testing.assert_almost_equal(result['F-statistic'], 4.846087862380137)
    np.testing.assert_almost_equal(result['Group DoF'], 2)
    np.testing.assert_almost_equal(result['Group Mean Squares'], 1.8831700000000007)
    np.testing.assert_almost_equal(result['Group Sum of Squares'], 3.7663400000000014)

    np.testing.assert_almost_equal(result['p-value'], 0.01590995832562292)
    np.testing.assert_almost_equal(result['Residual DoF'], 27)
    np.testing.assert_almost_equal(result['Residual Mean Squares'], 0.38859592592592596)
    np.testing.assert_almost_equal(result['Residual Sum of Squares'], 10.492090000000001)

    del plants['Unnamed: 0']

    ctrl = plants[plants['group'] == 'ctrl']['weight'].reset_index()
    del ctrl['index']
    ctrl.rename(columns={'weight': 'ctrl'}, inplace=True)

    trt1 = plants[plants['group'] == 'trt1']['weight'].reset_index()

    del trt1['index']
    trt1.rename(columns={'weight': 'trt1'}, inplace=True)

    trt2 = plants[plants['group'] == 'trt2']['weight'].reset_index()

    del trt2['index']
    trt2.rename(columns={'weight': 'trt2'}, inplace=True)

    anov2 = AnovaOneWay(ctrl, trt1, trt2)

    result2 = anov2.test_summary

    np.testing.assert_almost_equal(result2['F-statistic'], 4.846087862380138)
    np.testing.assert_almost_equal(result2['Group DoF'], 2)
    np.testing.assert_almost_equal(result2['Group Mean Squares'], 1.8831700000000007)
    np.testing.assert_almost_equal(result2['Group Sum of Squares'], 3.766340000000002)

    np.testing.assert_almost_equal(result2['p-value'], 0.01590995832562281)
    np.testing.assert_almost_equal(result2['Residual DoF'], 27)
    np.testing.assert_almost_equal(result2['Residual Mean Squares'], 0.38859592592592596)
    np.testing.assert_almost_equal(result2['Residual Sum of Squares'], 10.492090000000001)


def test_ManovaOneWay():
    dat = multivariate_test_data()

    dat_shape = dat.shape
    manov = ManovaOneWay(dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], group=dat[:, 0])

    result = manov.test_summary

    assert result['degrees of freedom']['Denominator Degrees of Freedom'] == dat_shape[0] - dat_shape[1] - 1
    assert result['degrees of freedom']['Numerator Degrees of Freedom'] == dat_shape[1]
    assert result['Analysis Performed'] == 'One-Way MANOVA'

    pillai = result['Pillai Statistic']
    roy = result['Roys Statistic']
    wilk = result['Wilks Lambda']
    hotelling = result['Hotellings T^2']

    np.testing.assert_almost_equal(pillai['Pillai Statistic'], 1.3054724154813995)
    np.testing.assert_almost_equal(pillai['Pillai F-value'], 4.069718325783225)
    np.testing.assert_almost_equal(pillai['Pillai p-value'], 0.004209350934305522)

    np.testing.assert_almost_equal(roy['Roys Statistic'], 1.87567111989616)

    np.testing.assert_almost_equal(wilk["Wilks Lambda F-value"], 4.936888039729538)
    np.testing.assert_almost_equal(wilk["Wilks Lambda"], 0.15400766733804414)
    np.testing.assert_almost_equal(wilk["Wilks Lambda p-value"], 0.001210290803741243)

    np.testing.assert_almost_equal(hotelling["Hotellings T^2 Statistic"], 2.921368304265692)
