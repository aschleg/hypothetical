import numpy as np
import pandas as pd


def build_des_mat(*args, group=None):
    # arg_shapes = []
    #
    # for a in args:
    #     arg_shapes.append(a.shape[1])
    #
    # if len(set(arg_shapes)) != 1:
    #     raise ValueError('all input arrays must be of the same dimension')
    #
    # if all(x == 1 for x in arg_shapes):

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
