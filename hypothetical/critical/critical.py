

from numpy import nan


def w_critical_value(n, alpha, alternative):

    if isinstance(n, str):
        n = int(n)
    if isinstance(alpha, str):
        alpha = float(alpha)

    if n > 30:
        raise ValueError('W critical values are only provided for n >= 30.')
    if alpha not in (0.05, 0.01):
        raise ValueError('W critical values are only provided for alpha values 0.05 and 0.01.')
    if alternative not in ('one-tail', 'two-tail'):
        raise ValueError("alternative must be one of 'one-tail' or 'two-tail'.")

    return w_critical_value_table[alternative][alpha][n]


# http://users.stat.ufl.edu/~winner/tables/wilcox_signrank.pdf
w_critical_value_table = {
    'one-tail': {
        0.01: {
            5: nan,
            6: nan,
            7: 0,
            8: 1,
            9: 3,
            10: 5,
            11: 7,
            12: 9,
            13: 12,
            14: 15,
            15: 19,
            16: 23,
            17: 27,
            18: 32,
            19: 37,
            20: 43,
            21: 49,
            22: 55,
            23: 62,
            24: 69,
            25: 76,
            26: 84,
            27: 92,
            28: 101,
            29: 110,
            30: 120
        },
        0.05: {
            5: 0,
            6: 2,
            7: 3,
            8: 5,
            9: 8,
            10: 10,
            11: 13,
            12: 17,
            13: 21,
            14: 25,
            15: 30,
            16: 35,
            17: 41,
            18: 47,
            19: 53,
            20: 60,
            21: 67,
            22: 75,
            23: 83,
            24: 91,
            25: 100,
            26: 110,
            27: 119,
            28: 130,
            29: 140,
            30: 151
        }
    },
    'two-tail': {
        0.01: {
            5: nan,
            6: nan,
            7: nan,
            8: 0,
            9: 1,
            10: 3,
            11: 5,
            12: 7,
            13: 9,
            14: 12,
            15: 15,
            16: 19,
            17: 23,
            18: 27,
            19: 32,
            20: 37,
            21: 42,
            22: 48,
            23: 54,
            24: 61,
            25: 68,
            26: 75,
            27: 83,
            28: 91,
            29: 100,
            30: 109
        },
        0.05: {
            5: nan,
            6: 0,
            7: 2,
            8: 3,
            9: 5,
            10: 8,
            11: 10,
            12: 13,
            13: 17,
            14: 21,
            15: 25,
            16: 29,
            17: 34,
            18: 40,
            19: 46,
            20: 52,
            21: 58,
            22: 65,
            23: 73,
            24: 81,
            25: 89,
            26: 98,
            27: 107,
            28: 116,
            29: 126,
            30: 137
        }
    }
}
