import numpy as np


def _build_des_mat(x, group=None, *args):

    if args is not ():
        c = args[0]
        for i in np.arange(1, len(args)):
            c = np.column_stack((c, args[i]))
        mat = np.column_stack((x, c))

    else:
        mat = x.copy()

    if mat.ndim > 1:
        mat = np.sum(mat, axis=1)

    if group is None:
        return mat

    data = np.column_stack([group, mat])

    return data
