import numpy as np
import pandas as pd


def build_des_mat(*args, group=None):
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


def build_summary_matrix(x, y=None):
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
