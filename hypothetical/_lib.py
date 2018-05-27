import numpy as np
import pandas as pd


def build_des_mat(*args, group=None):

    if group is None:
        c = pd.concat([*args], axis=1).melt()
    else:
        c = np.column_stack([*args])

        if group is not None:
            c = np.column_stack([group, c])

    if isinstance(c, pd.DataFrame):
        if c.shape[1] == 1:
            c = c.loc[:, c.columns[0]].values
        else:
            c = c.values

    return c
