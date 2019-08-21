import pytest
import numpy as np
import pandas as pd
from hypothetical._lib import build_des_mat


def test_array():
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


def test_build_design_matrix():
    dat = test_array()
    dat_df = pd.DataFrame(dat)

    des_mat = build_des_mat(dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], group=dat[:, 0])
    des_mat_df = build_des_mat(dat_df[1], dat_df[2], dat_df[3], dat_df[4], group=dat_df[0])

    des_mat_no_group = build_des_mat(dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4])

    des_mat_group_df = build_des_mat(dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], group=pd.DataFrame(dat[:, 0]))

    des_mat_group_df = build_des_mat(dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], group=pd.DataFrame(dat[:, 0]))

    assert isinstance(des_mat, np.ndarray)
    assert des_mat.shape == dat.shape

    assert isinstance(des_mat_df, np.ndarray)
    assert des_mat_df.shape == dat.shape

    assert isinstance(des_mat_no_group, np.ndarray)
    assert des_mat_no_group.shape[1] == 2

    assert isinstance(des_mat, np.ndarray)
    assert des_mat_group_df.shape == dat.shape


def test_build_matrix():
    arr1 = [4, 4, 5, 5, 3, 2, 5]
    arr2 = [2, 3, 3, 3, 3, 3, 3]
