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

Variance and Covariance
-----------------------

.. autosummary::
    :toctree: generated/

    covariance
    var
    std_dev

.. autosummary::
    :toctree: generated/

    Cov
    Variance

Other Functions
---------------

.. autosummary::
    :toctree: generated/

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

    See Also
    --------
    Cov : class containing several algorithm implementations for computing the covariance and covariance
        matrices.

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
    matrix = _build_matrix(x, y)

    pearson_corr = np.empty((matrix.shape[1], matrix.shape[1]))

    cov_matrix = covariance(matrix)

    for i in np.arange(cov_matrix.shape[0]):
        for j in np.arange(cov_matrix.shape[0]):
            pearson_corr[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])

    return pearson_corr


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
    matrix = _build_matrix(x, y)

    rank_matrix = matrix.copy()

    for i in np.arange(rank_matrix.shape[1]):
        rank_matrix[:, i] = rankdata(matrix[:, i], 'average')

    spearman_corr = pearson(rank_matrix)

    return spearman_corr


def var(x, method=None):
    r"""
    Front-end interface function for computing the variance of a sample
    or population.

    Parameters
    ----------
    x : array_like
        Accepts a numpy array, nested list, dictionary, or
        pandas DataFrame. The private function _create_array
        is called to create a copy of x as a numpy array.
    method : {'corrected_two_pass', 'textbook_one_pass', 'standard_two_pass', 'youngs_cramer'}, optional.
        Selects algorithm used to calculate variance. Default method is :code:`corrected_two_pass` which
        is generally more computationally stable than other algorithms (with the exception of youngs-cramer,
        perhaps).

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

    See Also
    --------
    std_dev : function for computing the standard deviation of an observation array.
    Variance : class containing implementations of several available algorithms for computing the
        variance of two sample vectors or a data matrix.

    """
    x = Variance(x)
    if method is None:
        v = getattr(x, x.method, None)
    else:
        if hasattr(x, method):
            v = getattr(x, method, x.method)
        else:
            raise ValueError('no method with name ' + str(method))

    return v()


def std_dev(x):
    r"""
    Calculates the standard deviation by simply taking the square
    root of the variance.

    Parameters
    ----------
    x : array_like
        Numpy ndarray, pandas DataFrame or Series, list, or list of lists representing a 1D or 2D array
        containing the variables and their respective observation vectors.

    Returns
    -------
    sd : numpy array or float
        The computed standard deviation.

    See Also
    --------
    var : function for computing the variance of an observation array.

    """
    v = var(x)
    sd = np.sqrt(v)

    return sd


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


class Cov(object):
    r"""
    Class object containing the covariance matrix algorithms used by the covar function. Meant to be
    a backend to the covar function and therefore is not meant to be called directly.

    Parameters
    ----------
    x : array-like
        A 2-D array containing the variables and observations to compute the covariance matrix.
    y : array-like, optional
        Optional second matrix of same dimensions as x to compute covariance between two separate matrices.

    Attributes
    ----------
    x : array-like
        The design matrix of the input data for performing the covariance computation.
    n : int
        Number of rows.
    m : int
        Number of columns.

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

    See Also
    --------
    covariance : function for computing the covariance of two sample observation vectors or a data matrix.
        The :code:`covariance` function is meant to be the interface to the :code:`Cov` class, which acts
        as an internal 'back-end'.

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


class Variance(object):
    r"""
    Class containing various algorithm method implementations for computing the
    variance of a sample. Please see the individual methods for more details on
    the specific algorithms.

    Parameters
    ----------
    x : array_like
        Accepts a list, nested list, numpy ndarray, or pandas DataFrame or
        pandas Series.

    Attributes
    ----------
    type : str
        Class type of object that initializes the class.
    dim : int
        The dimension of the array
    method : str
        The default method for calculating the variance.
    n : int
        Number of rows of the array

    Methods
    -------
    textbook_one_pass()
        So-called due the equation's prevalence in statistical textbooks (Chan, Golub, & Leveque, 1983).
    standard_two_pass()
        Known as two-pass as it passes through the data twice, once to compute the mean and
        again to compute the variance :math:`S`.
    corrected_two_pass()
        An alternative form of the standard two pass algorithm suggested by Professor Å. Björck.
    youngs_cramer()
        Implementation of the Youngs-Cramer updating algorithm for computing variance.

    See Also
    --------
    var : function for computing the variance of a sample observation vector. The :code:`var` function is meant
        to be the primary interface to the :code:`Variance` class.

    """
    def __init__(self, x):
        self.type = x.__class__.__name__
        if isinstance(x, pd.DataFrame):
            self.x = x.values
        elif isinstance(x, np.ndarray) is False:
            self.x = np.array(x)
        else:
            self.x = x

        if self.x.ndim > 2:
            raise ValueError('array must be 1D or 2D')

        self.dim = self.x.ndim
        self.n = self.x.shape[0]
        self.method = 'corrected_two_pass'

    def corrected_two_pass(self):
        r"""
        Computes variance using the corrected two pass algorithm as suggested by
        Professor Å. Björck in (Chan, Golub, & Leveque, 1983). The corrected two pass
        approach is generally more stable numerically compared to other methods and is
        the default algorithm used in the var function.

        Returns
        -------
        varr : numpy ndarray or float
            Depending on the dimension of the input, returns a 1D array of the
            column-wise computed variances or a float if given a 1D array.

        Notes
        -----
        The corrected two pass algorithm takes advantage of increased gains in accuracy by
        shifting all the data by the computed mean before computing :math:`S`. Even primitive
        approximations of :math:`\bar{x}` can yield large improvements in accuracy. The
        corrected two pass algorithm is defined as:

        .. math::

            S = \sum^N_{i=1} (x_i - \bar{x})^2 - \frac{1}{N} \left( \sum^N_{i=1} (x_i - \bar{x}) \right)^2

        The first term is the standard two pass algorithm while the second acts as an approximation
        to the error term of the first term that avoids the problem of catastrophic cancellation.

        References
        ----------
        Chan, T., Golub, G., & Leveque, R. (1983). Algorithms for Computing the Sample Variance:
            Analysis and Recommendations. The American Statistician, 37(3), 242-247.
            http://dx.doi.org/10.1080/00031305.1983.10483115

        """
        if self.dim == 1:
            varr = (np.sum(np.power(self.x - np.mean(self.x), 2)) - (1 / self.n) *
                    np.power(np.sum(self.x - np.mean(self.x)), 2)) / (self.n - 1)

        else:
            varr = np.empty(self.x.shape[1])
            j = 0
            for i in self.x.T:
                varr[j] = (np.sum(np.power(i - np.mean(i), 2)) - (1 / self.n) *
                           np.power(np.sum(i - np.mean(i)), 2)) / (self.n - 1)

                j += 1

        return varr

    def textbook_one_pass(self):
        r"""
        Textbook one-pass algorithm for calculating variance as defined in
        (Chan, Golub, & Leveque, 1983). Currently defined for 1D and 2D arrays.

        Returns
        -------
        varr : numpy ndarray or float
            Depending on the dimension of the input, returns a 1D array of the
            column-wise computed variances or a float if given a 1D array.

        Notes
        -----
        The textbook one pass algorithm for calculating variance is so named due to its
        prevalence in statistical textbooks and it passes through the data once
        (hence 'one-pass').

        The textbook one pass algorithm is defined as:

        .. math::

            S = \sum^N_{i=1} x_i^2 - \frac{1}{N}\left( \sum^N_{i=1} x_i \right)^2

        References
        ----------
        Chan, T., Golub, G., & Leveque, R. (1983). Algorithms for Computing the Sample Variance:
            Analysis and Recommendations. The American Statistician, 37(3), 242-247.
            http://dx.doi.org/10.1080/00031305.1983.10483115

        """
        if self.dim == 1:
            varr = (np.sum(np.power(self.x, 2.)) - (1. / self.n) *
                    np.power(np.sum(self.x), 2.)) / (self.n - 1)

        else:
            varr = np.empty(self.x.shape[1])
            j = 0
            for i in self.x.T:
                varr[j] = (np.sum(np.power(i, 2.)) - (1. / self.n) * np.power(np.sum(i), 2.)) / (self.n - 1)
                j += 1

        return varr

    def standard_two_pass(self):
        r"""
        Standard two-pass algorithm defined in (Chan, Golub, & Leveque, 1983) for
        computing variance of a 1D or 2D array.

        Returns
        -------
        varr : numpy ndarray or float
            Depending on the dimension of the input, returns a 1D array of the
            column-wise computed variances or a float if given a 1D array.

        Notes
        -----
        The standard two pass algorithm for computing variance as defined in
        (Chan, Golub, & Leveque, 1983) is so named due to the algorithm passing
        through the data twice, once to compute the mean :math:`\bar{x}` and again
        for the variance :math:`S`. The standard two pass algorithm is defined as:

        .. math::

            S = \sum^N_{i=1} (x_i - \bar{x})^2 \qquad \bar{x} = \frac{1}{N} \sum^N_{i=1} x_i

        Due to the algorithm's two pass nature, it may not be the most optimal approach
        when the data is too large to store in memory or dynamically as data is collected.
        The algorithm is mathematically equivalent to the textbook one-pass algorithm.

        References
        ----------
        Chan, T., Golub, G., & Leveque, R. (1983). Algorithms for Computing the Sample Variance:
            Analysis and Recommendations. The American Statistician, 37(3), 242-247.
            http://dx.doi.org/10.1080/00031305.1983.10483115

        """
        if self.dim == 1:
            varr = np.sum(np.power(self.x - np.mean(self.x), 2)) / (self.n - 1)

        else:
            varr = np.empty(self.x.shape[1])
            j = 0
            for i in self.x.T:
                varr[j] = np.sum(np.power(i - np.mean(i), 2)) / (self.n - 1)
                j += 1

        return varr

    def youngs_cramer(self):
        r"""
        Implementation of the Youngs-Cramer updating algorithm for computing the variance
        :math:`S` as presented in (Chan, Golub, & LeVeque, 1982).

        Returns
        -------
        varr : numpy ndarray or float
            Depending on the dimension of the input, returns a 1D array of the
            column-wise computed variances or a float if given a 1D array.

        Notes
        -----
        Updating algorithms for computing variance have been proposed by numerous authors as
        they are robust to catastrophic cancellation and don't require several passes through
        the data, hence reducing the amount of memory required. The Youngs and Cramer updating
        algorithm is generally as performant as the two-pass algorithm. The algorithm proposed by
        Youngs and Cramer follows from their investigation of the most performant updating
        algorithms for computing variance and is as follows:

        .. math::

            t_j = t_{j-1} + x_j
            S_n = S_{n-1} + \frac{1}{n(n - 1)} (nx_j - t_j)^2

        References
        ----------
        Chan, T., Golub, G., & Leveque, R. (1983). Algorithms for Computing the Sample Variance:
            Analysis and Recommendations. The American Statistician, 37(3), 242-247.
            http://dx.doi.org/10.1080/00031305.1983.10483115

        Chan, T., Golub, G., & LeVeque, R. (1982). Updating Formulae and a Pairwise Algorithm for
            Computing Sample Variances. COMPSTAT 1982 5Th Symposium Held At Toulouse 1982, 30-41.
            http://dx.doi.org/10.1007/978-3-642-51461-6_3

        """
        if self.dim == 1:
            s = 0
            n = 1
            t = self.x[0]

            for j in np.arange(1, self.n):
                n += 1
                t = t + self.x[j]
                s = s + (1. / (n * (n - 1)) * np.power(n * self.x[j] - t, 2))

            varr = s / float(self.n - 1)

        else:
            varr = np.empty(self.x.shape[1])
            k = 0

            for i in self.x.T:
                s = 0
                n = 1
                t = i[0]

                for j in np.arange(1, self.n):
                    n += 1
                    t = t + i[j]
                    s = s + (1. / (n * (n - 1))) * np.power(n * i[j] - t, 2)

                s = s / (self.n - 1.)
                varr[k] = s
                k += 1

        return varr


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
