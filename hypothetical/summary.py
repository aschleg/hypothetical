import numpy as np
import pandas as pd
from scipy.stats import rankdata


def covariance(x, y=None, method=None):
    r"""
    Computes the covariance matrix.

    Parameters
    ----------
    x : array-like
        A 2-D array containing the variables and observations to compute the covariance matrix.
    y : array-like, optional
        Optional second matrix of same dimensions as x to compute covariance between two separate matrices.
    method : {'naive', 'shifted_covariance', 'two_pass_covariance'}, optional
        Method to compute the covariance matrix. Algorithms include the naive computation, shifted
        covariance and two pass covariance. Of these, the two pass algorithm is the most
        numerically stable and therefore is the default method.

    Returns
    -------
    array-like
        The covariance matrix of the input data.

    Examples
    --------
    >>> h = [[16,4,8,4], [4,10,8,4], [8,8,12,10], [4,4,10,12]]
    >>> covariance(h)
    array([[ 32.        ,  -8.        ,  -2.66666667, -10.66666667],
       [ -8.        ,   9.        ,   0.33333333,  -3.66666667],
       [ -2.66666667,   0.33333333,   3.66666667,   6.33333333],
       [-10.66666667,  -3.66666667,   6.33333333,  17.        ]])
    >>> covariance(h, method='naive')
    array([[ 32.        ,  -8.        ,  -2.66666667, -10.66666667],
       [ -8.        ,   9.        ,   0.33333333,  -3.66666667],
       [ -2.66666667,   0.33333333,   3.66666667,   6.33333333],
       [-10.66666667,  -3.66666667,   6.33333333,  17.        ]])

    """
    if y is None:
        x = Cov(x)
    else:
        x = Cov(x, y)
    if method is None:
        v = getattr(x, x.method, None)
    else:
        if hasattr(x, method):
            v = getattr(x, method, x.method)
        else:
            raise ValueError('no method with name ' + str(method))

    return v()


def pearson(x, y=None):
    matrix = _build_matrix(x, y)

    pearson_corr = np.empty((matrix.shape[1], matrix.shape[1]))

    cov_matrix = covariance(matrix)

    for i in np.arange(cov_matrix.shape[0]):
        for j in np.arange(cov_matrix.shape[0]):
            pearson_corr[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])

    return pearson_corr


def spearman(x, y=None):
    matrix = _build_matrix(x, y)

    rank_matrix = matrix.copy()

    for i in np.arange(rank_matrix.shape[1]):
        rank_matrix[:, i] = rankdata(matrix[:, i], 'average')

    spearman_corr = pearson(rank_matrix)

    return spearman_corr


class Cov(object):
    r"""
    Class object containing the covariance matrix algorithms used by the covar function. Meant to be
    a backend to the covar function and therefore is not meant to be called directly.

    Methods
    -------
    naive()
        Implementation of the naive algorithm for estimating a covariance matrix.
    shifted_covariance()
        Implements the shifted covariance algorithm for computing a covariance matrix.
    two_pass_covariance()
        Estimates a covariance matrix using the two pass algorithm.

    Notes
    -----
    Covariance defines how two variables vary together. The elements :math:`i, j` of a covariance matrix
    :math:`C` is the covariance of the :math:`i`th and :math:`j`th elements of the random variables. More
    compactly, the covariance matrix extends the notion of variance to dimensions greater than 2. The
    covariance of two random variables :math:`X` and :math:`Y` is defined as the expected product of two
    variables' deviations from their respective expected values:

    .. math::

        cov(X, Y) = E[(X - E[X])(Y - E[Y])]

    Where :math:`E[X]` and :math:`E[Y]` are the expected values of the random variables :math:`X` and
    :math:`Y`, respectively, also known as the mean of the random variables.

    References
    ----------
    Algorithms for calculating variance. (2017, June 24). In Wikipedia, The Free Encyclopedia.
        From https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=787336827

    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    Weisstein, Eric W. "Covariance Matrix." From MathWorld--A Wolfram Web Resource.
        http://mathworld.wolfram.com/CovarianceMatrix.html

    """
    def __init__(self, x, y=None):
        self.x = _build_matrix(x, y)

        self.n, self.m = self.x.shape
        self.method = 'two_pass_covariance'
        self.cov = np.empty([self.m, self.m])

    def naive(self):
        r"""
        Implementation of the naive algorithm for estimating a covariance matrix.

        Returns
        -------
        array-like
            The estimated covariance matrix of the input data.

        Notes
        -----
        The naive algorithm for computing the covariance is defined as:

        .. math::

            Cov(X, Y) = \frac{\sum^n_{i=1} x_i y_i - (\sum^n_{i=1} x_i)(\sum^n_{i=1} y_i) / n}{n}

        References
        ----------
        Algorithms for calculating variance. (2017, June 24). In Wikipedia, The Free Encyclopedia.
            From https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=787336827

        """
        for i in np.arange(self.m):
            for j in np.arange(self.m):
                x, y = self.x[:, i], self.x[:, j]
                self.cov[i, j] = (np.sum(x * y) - np.sum(x) * np.sum(y) / self.n) / (self.n - 1)

        return self.cov

    def shifted_covariance(self):
        r"""
        Estimates a covariance matrix using the shifted covariance algorithm.

        Returns
        -------
        array-like
            The estimated covariance matrix of the input data.

        Notes
        -----
        The covariance of two random variables is shift-invariant (shift invariance defines that if a
        response :math:`y(n)` to an input :math:`x(n)`, then the response to an input :math:`x(n - k)`
        is :math:`y(n - k)`. Using the first values of each observation vector for their respective
        random variables as the shift values :math:`k_x` and :math:`k_y`, the algorithm can be defined as:

        .. math::

            Cov(X, Y) = Cov(X - k_x, Y - k_y) =
            \frac{\sum^n_{i=1}(x_i - k_x)(y_i - k_y) - (\sum^n^{i=1}(x_i - k_x))(\sum^n_{i=1}(y_i - k_y)) / n}{n}

        References
        ----------
        Algorithms for calculating variance. (2017, June 24). In Wikipedia, The Free Encyclopedia.
            From https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=787336827

        Shift-invariant system. (2017, June 30). In Wikipedia, The Free Encyclopedia.
            From https://en.wikipedia.org/w/index.php?title=Shift-invariant_system&oldid=788228439

        """
        for i in np.arange(self.m):
            for j in np.arange(self.m):
                x, y = self.x[:, i], self.x[:, j]
                kx = ky = x[0]
                self.cov[i, j] = (np.sum((x - kx) * (y - ky)) - np.sum(x - kx) * (np.sum(y - ky)) / self.n) / \
                                 (self.n - 1)

        return self.cov

    def two_pass_covariance(self):
        r"""
        Computes a covariance matrix by employing the two pass covariance algorithm. This algorithm is
        one of the more computationally stable algorithms for estimating a covariance matrix.

        Notes
        -----
        The two-pass covariance algorithm is another method that is generally more numerically stable
        in the computation of the covariance of two random variables as it first computes the sample means
        and then the covariance.

        First the sample means of the random variables:

        .. math::

            \bar{x} = \frac{1}{n} \sum^n_{i=1} x

            \bar{y} = \frac{1}{n} \sum^n_{i=1} y

        Then the covariance of the two variables is computed:

        .. math::

            Cov(X, Y) = \frac{\sum^n_{i=1}(x_i - \bar{x})(y_i - \bar{y})}{n}

        Returns
        -------
        array-like
            The estimated covariance matrix of the input data.

        References
        ----------
        Algorithms for calculating variance. (2017, June 24). In Wikipedia, The Free Encyclopedia.
            From https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=787336827

        """
        for i in np.arange(self.m):
            for j in np.arange(self.m):
                x, y = self.x[:, i], self.x[:, j]
                xbar, ybar = np.mean(x), np.mean(y)

                self.cov[i, j] = (np.sum((x - xbar) * (y - ybar))) / (self.n - 1)

        return self.cov


def _build_matrix(x, y=None):
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
