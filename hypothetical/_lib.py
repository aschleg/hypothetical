import numpy as np
import numpy_indexed as npi
import pandas as pd
from scipy.stats import rankdata


def _build_des_mat(*args, group=None):
    arg_list = []

    for arg in args:
        if isinstance(arg, pd.DataFrame):
            arg_list.append(arg.squeeze().values)
        else:
            arg_list.append(arg)

    if isinstance(group, pd.DataFrame):
        group = group.squeeze().values

    if group is None:
        c = pd.DataFrame(np.stack(arg_list, axis=-1)).melt()
    else:
        c = pd.concat([pd.DataFrame(np.vstack(arg_list))]).transpose()
        c.insert(0, 'group', group)

    if isinstance(c, pd.DataFrame):
        c = c.values

    return c


def _build_summary_matrix(x, y=None):
    if isinstance(x, pd.DataFrame):
        x = x.values
    elif not isinstance(x, np.ndarray):
        x = np.array(x)

    if y is not None:
        if isinstance(y, pd.DataFrame):
            y = y.values
        elif not isinstance(y, np.ndarray):
            y = np.array(y)

        x = np.column_stack([x, y])

    return x


def _rank(design_matrix):

    ranks = rankdata(design_matrix[:, 1], 'average')

    ranks = np.column_stack([design_matrix, ranks])

    return ranks


def _group_rank_sums(ranked_matrix):
    rank_sums = npi.group_by(ranked_matrix[:, 0],
                             ranked_matrix[:, 2],
                             np.sum)

    return rank_sums