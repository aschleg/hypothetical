# encoding=utf-8

"""
Functions for computing correlation and covariance of two variables or a data matrix. Several different
algorithms for the computation of covariance and variance are provided.

Correlation
-----------

.. autosummary::
    :toctree: generated/

    pearson
    spearman
    SimulateCorrelationMatrix

Variance and Covariance
-----------------------

.. autosummary::
    :toctree: generated/

    covar
    var
    std_dev

Other Functions
---------------

.. autosummary::
    :toctree: generated/

    add_noise
    kurtosis
    skewness
    mean_absolute_deviation
    variance_condition

References
----------
Algorithms for calculating variance. (2017, June 24). In Wikipedia, The Free Encyclopedia.
    From https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=787336827

Chan, T., Golub, G., & LeVeque, R. (1983). Algorithms for Computing the Sample Variance:
    Analysis and Recommendations. The American Statistician, 37(3), 242-247.
    http://dx.doi.org/10.1080/00031305.1983.10483115

Chan, T., Golub, G., & LeVeque, R. (1982). Updating Formulae and a Pairwise Algorithm for
    Computing Sample Variances. COMPSTAT 1982 5Th Symposium Held At Toulouse 1982, 30-41.
    http://dx.doi.org/10.1007/978-3-642-51461-6_3

Pearson correlation coefficient. (2017, July 12). In Wikipedia, The Free Encyclopedia.
    From https://en.wikipedia.org/w/index.php?title=Pearson_correlation_coefficient&oldid=790217169

Press, W., Teukolsky, S., Vetterling, W., & Flannery, B. (2007). Numerical recipes (3rd ed.).
    Cambridge: Cambridge University Press.

Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
    Brigham Young University: John Wiley & Sons, Inc.

Shift-invariant system. (2017, June 30). In Wikipedia, The Free Encyclopedia.
    From https://en.wikipedia.org/w/index.php?title=Shift-invariant_system&oldid=788228439

Spearman's rank correlation coefficient. (2017, June 24). In Wikipedia, The Free Encyclopedia.
    From https://en.wikipedia.org/w/index.php?title=Spearman%27s_rank_correlation_coefficient&oldid=787350680

Weisstein, Eric W. "Covariance Matrix." From MathWorld--A Wolfram Web Resource.
    http://mathworld.wolfram.com/CovarianceMatrix.html

"""


import numpy as np
import pandas as pd
from itertools import repeat
from scipy.stats import rankdata
from scipy.linalg import toeplitz

from hypothetical._lib import _build_summary_matrix


def add_noise(cor, epsilon=None, m=None):
    if isinstance(cor, pd.DataFrame):
        cor = cor.values
    elif isinstance(cor, np.ndarray) is False:
        cor = np.array(cor)

    n = cor.shape[1]

    if epsilon is None:
        epsilon = 0.05
    if m is None:
        m = 2

    np.fill_diagonal(cor, 1 - epsilon)

    cor = SimulateCorrelationMatrix._generate_noise(cor, n, m, epsilon)

    return cor


def covar(x, y=None, method=None):
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
    >>> covar(h)
    array([[ 32.        ,  -8.        ,  -2.66666667, -10.66666667],
       [ -8.        ,   9.        ,   0.33333333,  -3.66666667],
       [ -2.66666667,   0.33333333,   3.66666667,   6.33333333],
       [-10.66666667,  -3.66666667,   6.33333333,  17.        ]])
    >>> covar(h,method='naive')
    array([[ 32.        ,  -8.        ,  -2.66666667, -10.66666667],
       [ -8.        ,   9.        ,   0.33333333,  -3.66666667],
       [ -2.66666667,   0.33333333,   3.66666667,   6.33333333],
       [-10.66666667,  -3.66666667,   6.33333333,  17.        ]])

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

    **Available algorithms:**

    **Two Pass Covariance**

    The two-pass covariance algorithm is the default method as it is generally more numerically stable
    in the computation of the covariance of two random variables as it first computes the sample means
    and then the covariance.

    First the sample means of the random variables:

    .. math::

        \bar{x} = \frac{1}{n} \sum^n_{i=1} x

        \bar{y} = \frac{1}{n} \sum^n_{i=1} y

    Then the covariance of the two variables is computed:

    .. math::

        Cov(X, Y) = \frac{\sum^n_{i=1}(x_i - \bar{x})(y_i - \bar{y})}{n}

    **Naive Covariance**

    The naive algorithm for computing the covariance is defined as:

    .. math::

        Cov(X, Y) = \frac{\sum^n_{i=1} x_i y_i - (\sum^n_{i=1} x_i)(\sum^n_{i=1} y_i) / n}{n}

    **Shifted Covariance**

    The covariance of two random variables is shift-invariant (shift invariance defines that if a
    response :math:`y(n)` to an input :math:`x(n)`, then the response to an input :math:`x(n - k)`
    is :math:`y(n - k)`. Using the first values of each observation vector for their respective
    random variables as the shift values :math:`k_x` and :math:`k_y`, the shifted variance algorithm can be defined as:

    .. math::

        Cov(X, Y) = Cov(X - k_x, Y - k_y) =
        \frac{\sum^n_{i=1}(x_i - k_x)(y_i - k_y) - (\sum^n^{i=1}(x_i - k_x))(\sum^n_{i=1}(y_i - k_y)) / n}{n}

    References
    ----------
    Algorithms for calculating variance. (2017, June 24). In Wikipedia, The Free Encyclopedia.
        From https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=787336827

    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    Weisstein, Eric W. "Covariance Matrix." From MathWorld--A Wolfram Web Resource.
        http://mathworld.wolfram.com/CovarianceMatrix.html

    """
    x_mat = _build_summary_matrix(x, y)

    n, m = x_mat.shape
    cov = np.empty([m, m])

    if method is None or method == 'two-pass covariance':
        for i in np.arange(m):
            for j in np.arange(m):
                xx, yy = x_mat[:, i], x_mat[:, j]
                xbar, ybar = np.mean(xx), np.mean(yy)

                cov[i, j] = (np.sum((xx - xbar) * (yy - ybar))) / (n - 1)

    elif method == 'naive':
        for i in np.arange(m):
            for j in np.arange(m):
                x, y = x_mat[:, i], x_mat[:, j]
                cov[i, j] = (np.sum(x * y) - np.sum(x) * np.sum(y) / n) / (n - 1)

    elif method == 'shifted covariance':
        for i in np.arange(m):
            for j in np.arange(m):
                xx, yy = x_mat[:, i], x_mat[:, j]
                kx = ky = xx[0]
                cov[i, j] = (np.sum((xx - kx) * (yy - ky)) - np.sum(xx - kx) * (np.sum(yy - ky)) / n) / (n - 1)

    else:
        raise ValueError("method parameter must be one of 'two-pass covariance' (default), 'naive', "
                         "'shifted covariance, or None")

    return cov


def kurtosis(x, axis=0):
    r"""
    Computes the kurtosis of an array along a specified axis.

    Parameters
    ----------
    x : array-like
        One or two-dimensional array of data.
    axis : int {0, 1}
        Specifies which axis of the data to compute the kurtosis. The default is 0 (column-wise in a 2d-array). Cannot
        be greater than 1.

    Raises
    ------
    ValueError
        Raised if :code:`x` array is not a one or two-dimensional array.
    ValueError
        Raised if the :code:`axis` parameter is greater than 1.

    Returns
    -------
    k : float or array-like
        If x is one-dimensional, the kurtosis of the data is returned as a float. If x is two-dimensional, the
        calculated kurtosis along the specified axis is returned as a numpy array of floats.

    Examples
    --------
    >>> kurtosis([5, 2, 4, 5, 6, 2, 3])
    -1.4515532544378704
    >>> kurtosis([[5, 2, 4, 5, 6, 2, 3], [4, 6, 4, 3, 2, 6, 7]], axis=1)
    array([-1.45155325, -1.32230624])

    Notes
    -----
    Kurtosis, also known as the fourth moment, is non-dimensionl and measures the comparative 'flatness' or 'peak'
    of a given distribution to a normal distribution. Leptokurtic distributions have a positive kurtosis while
    platykurtic distributions have a negative kurtosis value. Though less commmon, distributions with a zero
    kurtosis value are called mesokurtic.

    Kurtosis is typically defined as:

    .. math::

        Kurt(x_0, \cdots, x_{n-1}) = \large{\frac{1}{n} \sum^{n-1}_{j=0} \large[\frac{x_j - \bar{x}}{\sigma}
        \large]^4 \large} - 3

    The :math:`-3` term is applied so a normal distribution will have a 0 kurtosis value (mesokurtic).

    References
    ----------
    Press, W., Teukolsky, S., Vetterling, W., & Flannery, B. (2007). Numerical recipes (3rd ed.).
        Cambridge: Cambridge University Press.

    Wikipedia contributors. (2018, August 28). Kurtosis. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:19, September 3, 2018, from https://en.wikipedia.org/w/index.php?title=Kurtosis&oldid=856893890

    """
    if axis > 1:
        raise ValueError('axis must be 0 (row-wise) or 1 (column-wise)')

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if x.ndim > 2:
        raise ValueError('array cannot have more than two dimensions')

    k = np.apply_along_axis(_kurt, axis, x)

    if k.shape == ():
        k = float(k)

    return k


def mean_absolute_deviation(x, axis=0, mean=False):
    r"""
    Calculates the mean absolute deviation of a data array along a specified axis.

    Parameters
    ----------
    x : array-like
        One or two-dimensional array of data.
    axis : {0, 1} int
        Specifies which axis of the data to compute the mean absolute deviation. The default is 0
        (column-wise in a 2d-array).
    mean : bool
        If False (default), the sample median is used to compute the mean absolute deviation. If True, the sample
        mean is used.

    Raises
    ------
    ValueError
        Raised if :code:`x` array is not a one or two-dimensional array.
    ValueError
        Raised if the :code:`axis` parameter is greater than 1.
    TypeError
        Raised if the :code:`mean` parameter is not boolean.

    Returns
    -------
    m : float or array-like
        If x is one-dimensional, the mean absolute deviation of the data is returned as a float. If x is
        two-dimensional, the calculated mean absolute deviation along the specified axis is returned as a numpy
        array of floats.

    Examples
    --------


    Notes
    -----


    References
    ----------
    Press, W., Teukolsky, S., Vetterling, W., & Flannery, B. (2007). Numerical recipes (3rd ed.).
        Cambridge: Cambridge University Press.

    """
    if axis > 1:
        raise ValueError('axis parameter must be 0 (row-wise) or 1 (column-wise)')

    if not isinstance(mean, bool):
        raise TypeError('mean parameter must be True or False.')

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if x.ndim > 2:
        raise ValueError('array cannot have more than two dimensions')

    if mean:
        m = np.apply_along_axis(_mad, axis, x)
    else:
        m = np.apply_along_axis(_med, axis, x)

    if m.shape == ():
        m = float(m)

    return m


def pearson(x, y=None):
    r"""
    Computes the Pearson product-moment correlation coefficients of the given variables.

    Parameters
    ----------
    x : array-like
        Numpy ndarray, pandas DataFrame or Series, list, or list of lists representing a 1D or 2D array
        containing the variables and their respective observation vectors. The input is concatenated with
        the parameter y if given.
    y : array-like, optional
        Numpy ndarray, pandas DataFrame or Series, list, or list of lists representing a 1D or 2D array
        containing the variables and their respective observation vectors.

    Returns
    -------
    numpy ndarray
        The Pearson product-moment correlation coefficient matrix of the inputted variables.

    Notes
    -----
    Pearson's product-moment correlation coefficient is the covariance of two random variables
    divided by the product of their standard deviations and is typically represented by
    :math:`\rho`:

    .. math::

        \rho_{x, y} = \frac{cov(X, Y)}{\sigma_X \sigma_Y}

    The correlation matrix :math:`C` and the covariance matrix :math:`R` have the following
    relationship.

    .. math::

        R_{ij} = \frac{C_{ij}}{\sqrt{C_{ii} * C_{jj}}}

    Examples
    --------
    >>> h = np.array([[16,4,8,4], [4,10,8,4], [8,8,12,10], [4,4,10,12]])
    >>> pearson(h)
    array([[ 1.        , -0.47140452, -0.24618298, -0.45732956],
       [-0.47140452,  1.        ,  0.05802589, -0.29643243],
       [-0.24618298,  0.05802589,  1.        ,  0.80218063],
       [-0.45732956, -0.29643243,  0.80218063,  1.        ]])
    >>> pearson(h[:, 0:1], h[:, 1:])
    array([[ 1.        , -0.47140452, -0.24618298, -0.45732956],
       [-0.47140452,  1.        ,  0.05802589, -0.29643243],
       [-0.24618298,  0.05802589,  1.        ,  0.80218063],
       [-0.45732956, -0.29643243,  0.80218063,  1.        ]])
    >>> pearson(h[:, 1], h[:, 2])
    array([[ 1.        ,  0.05802589],
       [ 0.05802589,  1.        ]])

    See Also
    --------
    spearman : function for computing the Spearman rank correlation of two vectors or a data matrix.

    References
    ----------
    Pearson correlation coefficient. (2017, July 12). In Wikipedia, The Free Encyclopedia.
        From https://en.wikipedia.org/w/index.php?title=Pearson_correlation_coefficient&oldid=790217169

    Rencher, A. (n.d.). Methods of Multivariate Analysis (2nd ed.).
        Brigham Young University: John Wiley & Sons, Inc.

    """
    matrix = _build_summary_matrix(x, y)

    pearson_corr = np.empty((matrix.shape[1], matrix.shape[1]))

    cov_matrix = covar(matrix)

    for i in np.arange(cov_matrix.shape[0]):
        for j in np.arange(cov_matrix.shape[0]):
            pearson_corr[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])

    return pearson_corr


class SimulateCorrelationMatrix(object):

    def __init__(self, k=None, nk=None, rho=None, M=None, power=None):

        if k is None:
            self.k = np.random.randint(3, 10)
        else:
            self.k = k
        if M is None:
            self.M = np.random.randint(1, 4)
        else:
            self.M = M
        if nk is None:
            self.nk = np.random.randint(2, 5, self.k)
        else:
            self.nk = nk
        if rho is None:
            self.rho = np.random.rand(self.k)
        else:
            self.rho = rho
        if power is None:
            self.power = 1
        else:
            self.power = power

        self.nkdim = int(np.sum(self.nk))
        self.method = 'constant'

    def constant(self):
        delta = np.min(self.rho) - 0.01
        cormat = np.full((self.nkdim, self.nkdim), delta)

        epsilon = 0.99 - np.max(self.rho)
        for i in np.arange(self.k):
            cor = np.full((self.nk[i], self.nk[i]), self.rho[i])

            if i == 0:
                cormat[0:self.nk[0], 0:self.nk[0]] = cor
            if i != 0:
                cormat[np.sum(self.nk[0:i]):np.sum(self.nk[0:i + 1]),
                np.sum(self.nk[0:i]):np.sum(self.nk[0:i + 1])] = cor

        np.fill_diagonal(cormat, 1 - epsilon)

        cormat = self._generate_noise(cormat, self.nkdim, self.M, epsilon)

        return cormat

    def toepz(self):
        cormat = np.zeros((self.nkdim, self.nkdim))

        epsilon = (1 - np.max(self.rho)) / (1 + np.max(self.rho)) - .01

        for i in np.arange(self.k):
            t = np.insert(np.power(self.rho[i], np.arange(1, self.nk[i])), 0, 1)
            cor = toeplitz(t)
            if i == 0:
                cormat[0:self.nk[0], 0:self.nk[0]] = cor
            if i != 0:
                cormat[np.sum(self.nk[0:i]):np.sum(self.nk[0:i + 1]),
                np.sum(self.nk[0:i]):np.sum(self.nk[0:i + 1])] = cor

        np.fill_diagonal(cormat, 1 - epsilon)

        cormat = self._generate_noise(cormat, self.nkdim, self.M, epsilon)

        return cormat

    def hub(self):
        cormat = np.zeros((self.nkdim, self.nkdim))

        for i in np.arange(self.k):
            cor = toeplitz(self._fill_hub_matrix(self.rho[i, 0], self.rho[i, 1], self.power, self.nk[i]))
            if i == 0:
                cormat[0:self.nk[0], 0:self.nk[0]] = cor
            if i != 0:
                cormat[np.sum(self.nk[0:i]):np.sum(self.nk[0:i + 1]),
                np.sum(self.nk[0:i]):np.sum(self.nk[0:i + 1])] = cor
            tau = (np.max(self.rho[i]) - np.min(self.rho[i])) / (self.nk[i] - 2)

        epsilon = 0.08 #(1 - np.min(rho) - 0.75 * np.min(tau)) - 0.01

        np.fill_diagonal(cormat, 1 - epsilon)

        cormat = self._generate_noise(cormat, self.nkdim, self.M, epsilon)

        return cormat

    @staticmethod
    def _generate_noise(cormat, N, M, epsilon):
        ev = []
        for _ in repeat(None, N):
            ei = np.random.uniform(low=-1, high=1, size=M)
            ev.append(np.sqrt(epsilon) * ei / np.sqrt(np.sum(np.power(ei, 2))))

        ev = np.array(ev).T
        E = np.dot(ev.T, ev)
        cormat = cormat + E

        return cormat

    @staticmethod
    def _fill_hub_matrix(rmax, rmin, power, p):
        rho = np.empty(p)
        rho[0] = 1
        for i in np.arange(1, p):
            rho[i] = rmax - np.power((i - 1) / (p - 1), power * (rmax - rmin))

        return rho


def skewness(x, axis=0):
    r"""
    Calculates the skewness of a given array.

    Parameters
    ----------
    x : array-like
        One or two-dimensional array of data.
    axis : int {0, 1}
        Specifies which axis of the data to compute the skewness. The default is 0 (column-wise in a 2d-array). Cannot
        be greater than 1.

    Raises
    ------
    ValueError
        Raised if :code:`x` array is not a one or two-dimensional array.
    ValueError
        Raised if the :code:`axis` parameter is greater than 1.

    Returns
    -------
    s : float or array-like
        If the given array is one-dimensional, a float value is returned. If two dimensional, an array is returned
        with the calculated skewness across the given axis.

    Examples
    --------
    >>> skewness([5, 2, 4, 5, 6, 2, 3])
    -0.028285981029545847
    >>> skewness([[5, 2, 4, 5, 6, 2, 3], [4, 6, 4, 3, 2, 6, 7]], axis=1)
    array([-0.02828598, -0.03331004])

    Notes
    -----
    The skewness, also known as the third moment, measures the degree of asymmetry of the given distribution around
    its mean. Skewness is typically defined as:

    .. math::

        Skew (x_0, \cdots, x_{n-1}) = \frac{1}{n} \sum^{n-1}_{j=0} \large[\frac{x_j - \bar{x}}{\sigma} \large]^3

    A positive skewness signifies an asymmetric distribution with a tail extending in the positive direction of x,
    whereas a negative skewness denotes an asymmetric distribution with the tail extending towards negative x.

    References
    ----------
    Press, W., Teukolsky, S., Vetterling, W., & Flannery, B. (2007). Numerical recipes (3rd ed.).
        Cambridge: Cambridge University Press.

    Wikipedia contributors. (2018, August 13). Skewness. In Wikipedia, The Free Encyclopedia.
        Retrieved 12:18, September 3, 2018, from https://en.wikipedia.org/w/index.php?title=Skewness&oldid=854777849

    """
    if axis > 1:
        raise ValueError('axis must be 0 (row-wise) or 1 (column-wise)')

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if x.ndim > 2:
        raise ValueError('array cannot have more than two dimensions')

    s = np.apply_along_axis(_skew, axis, x)

    if s.shape == ():
        s = float(s)

    return s


def spearman(x, y=None):
    r"""
    Computes the Spearman correlation coefficients of the given variables.

    Parameters
    ----------
    x : array-like
        Numpy ndarray, pandas DataFrame or Series, list, or list of lists representing a 1D or 2D array
        containing the variables and their respective observation vectors. The input is concatenated with
        the parameter y if given.
    y : array-like, optional
        Numpy ndarray, pandas DataFrame or Series, list, or list of lists representing a 1D or 2D array
        containing the variables and their respective observation vectors.

    Returns
    -------
    numpy ndarray
        The correlation coefficient matrix of the inputted variables.

    Notes
    -----
    Spearman's :math:`\rho`, often denoted :math:`r_s` is a nonparametric measure of correlation.
    While Pearson's product-moment correlation coefficient represents the linear relationship between
    two variables, Spearman's correlation measures the monotonicity of two variables. Put more simply,
    Spearman's correlation is Pearson's correlation performed on ranked variables.

    Two random variables :math:`X` and :math:`Y` and their respective observation vectors
    :math:`x_1, x_2, \cdots, x_n` and :math:`y_1, y_2, \cdots, y_n` are converted to ranked variables
    (identical values are averaged), often denoted :math:`rg_X` and :math:`rg_Y`, and the correlation
    :math:`r_s` is computed as:

    .. math::

        r_s = \rho_{rg_X, rg_Y} = \frac{cov(rg_X, rg_Y}{\sigma_{rg_X} \sigma_{rg_Y}}

    Where :math:`\rho` is the Pearson correlation coefficient applied to the ranked variables,
    :math:`cov(rg_X, rg_Y)` is the covariance of the ranked variables and :math:`\sigma_{rg_X}` and
    :math:`\sigma_{rg_Y}` are the standard deviations of the ranked variables.

    Examples
    --------
    >>> h = np.array([[16,4,8,4], [4,10,8,4], [8,8,12,10], [4,4,10,12]])
    >>> spearman(h)
    array([[ 1.        , -0.33333333, -0.03703704, -0.33333333],
       [-0.33333333,  1.        , -0.03703704, -0.33333333],
       [-0.03703704, -0.03703704,  1.        ,  0.85185185],
       [-0.33333333, -0.33333333,  0.85185185,  1.        ]])
    >>> spearman(h[:, 0:1], h[:, 1:])
    array([[ 1.        , -0.33333333, -0.03703704, -0.33333333],
       [-0.33333333,  1.        , -0.03703704, -0.33333333],
       [-0.03703704, -0.03703704,  1.        ,  0.85185185],
       [-0.33333333, -0.33333333,  0.85185185,  1.        ]])
    >>> spearman(h[:, 0], h[:, 1])
    array([[ 1.        , -0.33333333],
       [-0.33333333,  1.        ]])

    See Also
    --------
    pearson : function for computing the Pearson product-moment correlation of two vectors or a data matrix.

    References
    ----------
    Spearman's rank correlation coefficient. (2017, June 24). In Wikipedia, The Free Encyclopedia.
        From https://en.wikipedia.org/w/index.php?title=Spearman%27s_rank_correlation_coefficient&oldid=787350680

    """
    matrix = _build_summary_matrix(x, y)

    rank_matrix = matrix.copy()

    for i in np.arange(rank_matrix.shape[1]):
        rank_matrix[:, i] = rankdata(matrix[:, i], 'average')

    spearman_corr = pearson(rank_matrix)

    return spearman_corr


def std_dev(x):
    r"""
    Calculates the standard deviation by taking the square root of the variance.

    Parameters
    ----------
    x : array_like
        Numpy ndarray, pandas DataFrame or Series, list, or list of lists representing a 1D or 2D array
        containing the variables and their respective observation vectors.

    Returns
    -------
    sd : numpy array or float
        The computed standard deviation.

    Examples
    --------
    >>> s = std_dev([2, 5])
    2.12132034
    >>> s2 = std_dev([[5, 2], [2, 5]])
    array([2.12132034, 2.12132034])

    Notes
    -----
    The standard deviation is defined as the square root of the variance. For example, the corrected two pass
    algorithm for computing variance is defined as:

    .. math::

        S = \sum^N_{i=1} (x_i - \bar{x})^2 - \frac{1}{N} \left( \sum^N_{i=1} (x_i - \bar{x}) \right)^2

    Thus, the standard deviation would be defined as:

    .. math::

        \sigma = \sqrt{S}

    See Also
    --------
    var : function for computing the variance of an observation array.

    """
    v = var(x)
    sd = np.sqrt(v)

    return sd


def var(x, method='corrected two pass', correction=True):
    r"""
    Front-end interface function for computing the variance of a sample
    or population.

    Parameters
    ----------
    x : array_like
        One or two-dimensional array of data points. Accepts a numpy array, list, pandas DataFrame, or pandas Series.
    method : {'corrected_two_pass', 'textbook_one_pass', 'standard_two_pass', 'youngs_cramer'}, optional.
        Selects algorithm used to calculate variance. Default method is :code:`corrected_two_pass` which
        is generally more computationally stable than other algorithms (with the exception of youngs-cramer,
        perhaps).
    correction : bool
        If True (default), Bessel's correction, :math:`n - 1` is used in computing the variance rather than
        :math:`n`.

    Returns
    -------
    v : float or numpy array or numpy structured array or pandas DataFrame
        If the input is one-dimensional, the variance is returned as
        a float. For a two-dimensional input, the variance is calculated
        column-wise and returned as a numpy array or pandas DataFrame.

    Examples
    --------
    >>> f = pd.DataFrame({0: [1,-1,2,2], 1: [-1,2,1,-1], 2: [2,1,3,2], 3: [2,-1,2,1]})
    >>> var(f)
    np.array([2, 2.25, 0.666667, 2])
    >>> var(f[1])
    np.array([2])

    Notes
    -----

    ** Available algorithms**

    **Corrected Two Pass**

    The corrected two pass approach, as suggested by Professor Å. Björck in (Chan, Golub, & Leveque, 1983)
    is generally more stable numerically compared to other methods and is the default algorithm used in
    the var function.

    The corrected two pass algorithm takes advantage of increased gains in accuracy by
    shifting all the data by the computed mean before computing :math:`S`. Even primitive
    approximations of :math:`\bar{x}` can yield large improvements in accuracy. The
    corrected two pass algorithm is defined as:

    .. math::

        S = \sum^N_{i=1} (x_i - \bar{x})^2 - \frac{1}{N} \left( \sum^N_{i=1} (x_i - \bar{x}) \right)^2

    The first term is the standard two pass algorithm while the second acts as an approximation
    to the error term of the first term that avoids the problem of catastrophic cancellation.

    **Textbook One-Pass**

    The textbook one pass algorithm for calculating variance is so named due to its
    prevalence in statistical textbooks and it passes through the data once
    (hence 'one-pass').

    The textbook one pass algorithm is defined as:

    .. math::

        S = \sum^N_{i=1} x_i^2 - \frac{1}{N}\left( \sum^N_{i=1} x_i \right)^2

    **Standard Two-Pass**

    Standard two-pass algorithm defined in (Chan, Golub, & Leveque, 1983) for
    computing variance of a 1D or 2D array.

    The standard two pass algorithm for computing variance as defined in
    (Chan, Golub, & Leveque, 1983) is so named due to the algorithm passing
    through the data twice, once to compute the mean :math:`\bar{x}` and again
    for the variance :math:`S`. The standard two pass algorithm is defined as:

    .. math::

        S = \sum^N_{i=1} (x_i - \bar{x})^2 \qquad \bar{x} = \frac{1}{N} \sum^N_{i=1} x_i

    Due to the algorithm's two pass nature, it may not be the most optimal approach
    when the data is too large to store in memory or dynamically as data is collected.
    The algorithm is mathematically equivalent to the textbook one-pass algorithm.

    **Youngs-Cramer**

    Implementation of the Youngs-Cramer updating algorithm for computing the variance
    :math:`S` as presented in (Chan, Golub, & LeVeque, 1982).

    Updating algorithms for computing variance have been proposed by numerous authors as
    they are robust to catastrophic cancellation and don't require several passes through
    the data, hence reducing the amount of memory required. The Youngs and Cramer updating
    algorithm is generally as performant as the two-pass algorithm. The algorithm proposed by
    Youngs and Cramer follows from their investigation of the most performant updating
    algorithms for computing variance and is as follows:

    .. math::

        t_j = t_{j-1} + x_j
        S_n = S_{n-1} + \frac{1}{n(n - 1)} (nx_j - t_j)^2

    See Also
    --------
    std_dev : function for computing the standard deviation of an observation array.

    References
    ----------
    Chan, T., Golub, G., & Leveque, R. (1983). Algorithms for Computing the Sample Variance:
        Analysis and Recommendations. The American Statistician, 37(3), 242-247.
        http://dx.doi.org/10.1080/00031305.1983.10483115

    Press, W., Teukolsky, S., Vetterling, W., & Flannery, B. (2007). Numerical recipes (3rd ed.).
        Cambridge: Cambridge University Press.

    """

    if isinstance(x, pd.DataFrame):
        xx = x.values
    elif isinstance(x, np.ndarray) is False:
        xx = np.array(x)
    else:
        xx = x

    if xx.ndim > 2:
        raise ValueError('array must be 1D or 2D')

    dim = xx.ndim
    n = xx.shape[0]

    if correction:
        d = (n - 1)
    else:
        d = n

    if method is None or method == 'corrected two pass':
        if dim == 1:
            varr = (np.sum(np.power(xx - np.mean(xx), 2)) - (1 / n) *
                    np.power(np.sum(xx - np.mean(xx)), 2)) / d

        else:
            varr = np.empty(xx.shape[1])
            j = 0
            for i in xx.T:
                varr[j] = (np.sum(np.power(i - np.mean(i), 2)) - (1 / n) *
                           np.power(np.sum(i - np.mean(i)), 2)) / d

                j += 1

    elif method == 'textbook one pass':
        if dim == 1:
            varr = (np.sum(np.power(xx, 2.)) - (1. / n) *
                    np.power(np.sum(xx), 2.)) / d

        else:
            varr = np.empty(xx.shape[1])
            j = 0
            for i in xx.T:
                varr[j] = (np.sum(np.power(i, 2.)) - (1. / n) * np.power(np.sum(i), 2.)) / d
                j += 1

    elif method == 'standard two pass':
        if dim == 1:
            varr = np.sum(np.power(xx - np.mean(xx), 2)) / d

        else:
            varr = np.empty(xx.shape[1])
            j = 0
            for i in xx.T:
                varr[j] = np.sum(np.power(i - np.mean(i), 2)) / d
                j += 1

    elif method == 'youngs cramer':
        if dim == 1:
            s = 0
            nn = 1
            t = xx[0]

            for j in np.arange(1, n):
                nn += 1
                t = t + xx[j]
                s = s + (1. / (nn * (nn - 1)) * np.power(nn * xx[j] - t, 2))

            varr = s / float(d)

        else:
            varr = np.empty(xx.shape[1])
            k = 0

            for i in xx.T:
                s = 0
                nn = 1
                t = i[0]

                for j in np.arange(1, n):
                    nn += 1
                    t = t + i[j]
                    s = s + (1. / (nn * (nn - 1))) * np.power(nn * i[j] - t, 2)

                s = s / d
                varr[k] = s
                k += 1

    else:
        raise ValueError("method parameter must be one of 'corrected two pass' (default), 'textbook one pass', "
                         "'standard two pass', 'youngs cramer', or None.")

    return varr


def variance_condition(x):
    r"""
    Calculates the condition number, denoted as :math:`\kappa` which
    measures the sensitivity of the variance :math:`S` of a sample
    vector :math:`x` as defined by Chan and Lewis (as cited in Chan,
    Golub, & Leveque, 1983). Given a machine accuracy value of
    :math:`u`, the value :math:`\kappa u` can be used as a measure to
    judge the accuracy of the different variance computation algorithms.

    Parameters
    ----------
    x : array_like
        Numpy ndarray, pandas DataFrame or Series, list, or list of lists representing a 1D or 2D array
        containing the variables and their respective observation vectors.

    Returns
    -------
    varr : numpy ndarray
        Depending on the dimension of the input, returns a 1D or 2D array of the
        column-wise computed variances.

    Notes
    -----
    The 2-norm is defined as usual:

    .. math::

        ||x||_2 = \sum^N_{i=1} x^2_i

    Then the condition number :math:`\kappa` is defined as:

    .. math::

        \kappa = \frac{||x||_2}{\sqrt{S}} = \sqrt{1 + \bar{x}^2 N / S}

    References
    ----------
    Chan, T., Golub, G., & Leveque, R. (1983). Algorithms for Computing the Sample Variance:
        Analysis and Recommendations. The American Statistician, 37(3), 242-247.
        http://dx.doi.org/10.1080/00031305.1983.10483115

    """
    if isinstance(x, pd.DataFrame):
        x = x.values
    elif isinstance(x, np.ndarray) is False:
        x = np.array(x)

    if x.ndim == 1:
        kap_cond = np.linalg.norm(x) / std_dev(x)

    elif x.ndim == 2:
        kap_cond = np.empty(x.shape[1])
        j = 0
        for i in x.T:
            k = np.linalg.norm(i) / std_dev(i)
            kap_cond[j] = k
            j += 1

    else:
        raise ValueError('array must be 1D or 2D')

    return kap_cond


def _kurt(x, normal=True):
    n = x.shape[0]
    m = np.mean(x)

    kurt = np.sum(((x - m) ** 4. / n) / np.sqrt(var(x, correction=False)) ** 4.) - (3. * normal)

    return kurt


def _skew(x):
    n = x.shape[0]
    m = np.mean(x)

    skew = np.sum(((x - m) ** 3. / n) / np.sqrt(var(x, correction=False)) ** 3.)

    return skew


def _mad(x):
    n = x.shape[0]
    m = np.mean(x)

    mad = (1. / n) * np.sum(np.absolute(x - m))

    return mad


def _med(x):
    n = x.shape[0]
    m = np.median(x)

    med = (1. / n) * np.sum(np.absolute(x - m))

    return med
