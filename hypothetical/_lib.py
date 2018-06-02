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

    if isinstance(group, pd.DataFrame):
        group = group.squeeze().values

    if group is None:
        c = pd.DataFrame(np.stack(args, axis=-1)).melt()
    else:
        c = pd.concat([pd.DataFrame(np.vstack(args))]).transpose()
        c.insert(0, 'group', group)

    if isinstance(c, pd.DataFrame):
            c = c.values

    return c
