import numpy as np
from hypothetical.descriptive import covar, pearson


class FactorAnalysis(object):
    r"""

    Parameters
    ----------
    group_sample1, group_sample2, ... : array-like
        Corresponding observation vectors of the group samples. Must be the same length
        as the group parameter. If the group parameter is None, each observation vector
        will be treated as a group sample vector.
    factors : int, default None
        Number of underlying hypothetical factors
    rotate : str, default None
        Rotation to use when performing the factor analysis. Currently not used.
    covar : boolean, default False
        If False (default), perform the factor analysis using the covariance matrix. If
        True, the factor analysis is computed with the correlation matrix. It is highly
        recommended to use the correlation matrix in the vast majority of cases as
        variables with comparatively large variances can dominate the diagonal of the
        covariance matrix and the factors.

    """
    def __init__(self, x, factors=2, rotate=None, covar=False, method='principal_component'):
        r"""
        Initializes the FactorAnalysis class.

        Parameters
        ----------
        x : array-like
        Numpy ndarray, pandas DataFrame, list of lists or dictionary (keys are column
        names and corresponding values are the column values) representing observation
        vectors
    factors : int, default 2
        The hypothetical number of factors to fit.
        rotate : str, default None
            Rotation to use when performing the factor analysis. Currently not used.
        covar : boolean, default False
            If False (default), perform the factor analysis using the covariance matrix. If
            True, the factor analysis is computed with the correlation matrix. It is highly
            recommended to use the correlation matrix in the vast majority of cases as
            variables with comparatively large variances can dominate the diagonal of the
            covariance matrix and the factors.

        """
        if method not in ('principal_component', 'principal_factor', 'iterated_principal_factor'):
            raise ValueError('method parameter must be one of "principal_component" (default), "principal_factor", or '
                             '"iterated_principal_factor".')
        self.x = x
        self.factors = int(factors)

        if self.factors > self.x.shape[1]:
            raise ValueError('number of factors cannot exceed number of observation vectors')

        self.rotate = rotate
        self.covar = covar
        self.method = method

        if self.method == 'principal_component':
            self.loadings, self.h2, self.u2, self.com, self.proportion_loadings, self.var_proportion, \
                self.exp_proportion = self._principal_component()

        if self.method == 'principal_factor':
            self.loadings, self.h2, self.u2, self.com, self.proportion_loadings, self.var_proportion, \
                self.exp_proportion = self._principal_factor()

        if self.method == 'iterated_principal_factor':
            self.loadings, self.h2, self.u2, self.com, self.proportion_loadings, self.var_proportion, \
                self.exp_proportion, self.iterations = self._iterated_principal_factor()

        self.fa_results = {
            'h2': self.h2,
            'u2': self.u2,
            'com': self.com,
            'loadings': self.loadings,
            'proportion_loadings': self.proportion_loadings,
            'proportion_explained': self.exp_proportion,
            'proportion_variance': self.var_proportion,
            'method': self.method
        }

        if self.method == 'iterated_principal_factor':
            self.fa_results['iterations'] = self.iterations

    def _principal_component(self):
        r"""
        Performs factor analysis using the principal component method

        Returns
        -------
        namedtuple
            The factor analysis results are collected into a namedtuple with the following values:
            Factor Loadings
            Communality
            Specific Variance
            Complexity
            Proportion of Loadings
            Proportion of Variance
            Proportion of Variance Explained

        Notes
        -----
        The principal component method is rather misleading in its naming it that no principal
        components are calculated. The approach of the principal component method is to
        calculate the sample covariance matrix :math:`S` from a sample of data and then find an estimator,
        denoted :math:`\hat{\Lambda}` that can be used to factor :math:`S`.

        .. math::

            S = \hat{\Lambda} \hat{\Lambda}'

        Another term, :math:`\Psi`, is added to the estimate of :math:`S`, making the above
        :math:`S = \hat{\Lambda} \hat{\Lambda}' + \hat{\Psi}`. :math:`\hat{\Psi}` is a diagonal
        matrix of the specific variances :math:`(\hat{\psi_1}, \hat{\psi_2}, \cdots, \hat{\psi_p})`.
        :math:`\Psi` is estimated in other approaches to factor analysis such as the principal
        factor method and its iterated version but is excluded in the principal component method
        of factor analysis. The reason for the term's exclusion is since $\hat{\Psi}$ equals the
        specific variances of the variables, it models the diagonal of :math:`S` exactly.

        Spectral decomposition is employed to factor :math:`S` into:

        .. math::

            S = CDC'

        Where :math:`C` is an orthogonal matrix of the normalized eigenvectors of :math:`S` as
        columns and :math:`D` is a diagonal matrix with the diagonal equaling the eigenvalues
        of :math:`S`. Recall that all covariance matrices are positive semidefinite. Thus the
        eigenvalues must be either positive or zero which allows us to factor the diagonal matrix
        :math:`D` into:

        .. math::

            D = D^{1/2} D^{1/2}

        The above factor of :math:`D` is substituted into the decomposition of :math:`S`.

        .. math::

            S = CDC' = C D^{1/2} D^{1/2} C'

        Then rearranging:

        .. math::

            S = (CD^{1/2})(CD^{1/2})'

        Which yields the form :math:`S = \hat{\Lambda} \hat{\Lambda}'`. Since we are interested
        in finding :math:`m` factors in the data, we want to find a :math:`\hat{\Lambda}` that
        is :math:`p \times m` with :math:`m` smaller than :math:`p`. Thus :math:`D` can be
        defined as a diagonal matrix with :math:`m` eigenvalues (making it :math:`m \times m`) on
        the diagonal and :math:`C` is therefore :math:`p \times m` with the corresponding eigenvectors,
        which makes :math:`\hat{\Lambda} p \times m`.

        Note if the correlation matrix is used rather than the covariance matrix, there is no need
        to decompose the matrix in order to compute the eigenvalues and eigenvectors as correlation
        matrices are inherently positive semidefinite.

        References
        ----------
        Rencher, A. (2002). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        if self.covar is True:
            s = covar(self.x)
        else:
            s = pearson(self.x)

        eigvals, loadings, h2, u2, com = self._compute_factors(s)

        proportion_loadings, var_proportion, exp_proportion = self._compute_proportions(loadings, eigvals)

        return loadings, h2, u2, com, proportion_loadings, var_proportion, exp_proportion

    def _principal_factor(self):
        r"""
        Calculates the factor analysis with the principal factor (principal axis) method.

        Returns
        -------
        namedtuple
            The factor analysis results are collected into a namedtuple with the following values:
            Factor Loadings
            Communality
            Specific Variance
            Complexity
            Proportion of Loadings
            Proportion of Variance
            Proportion of Variance Explained

        Notes
        -----
        The principal factor method of factor analysis (also called the principal axis method)
        finds an initial estimate of :math:`\hat{\Psi}` and factors :math:`S - \hat{\Psi}`, or
        :math:`R - \hat{\Psi}` for the correlation matrix. Rearranging the estimated covariance
        and correlation matrices with the estimated :math:`p \times m` :math:`\hat{\Lambda}` matrix yields:

        .. math::

            S - \hat{\Psi} = \hat{\Lambda} \hat{\Lambda}^\prime
            R - \hat{\Psi} = \hat{\Lambda} \hat{\Lambda}^\prime

        Therefore the principal factor method begins with eigenvalues and eigenvectors of :math:`S - \hat{\Psi}`
        or :math:`R - \hat{\Psi}`. :math:`\hat{\Psi}` is a diagonal matrix of the :math:`i`th communality.
        As in the principal component method, the :math:`i`th communality, :math:`\hat{h}^2_i`, is equal to
        :math:`s_{ii} - \hat{\psi}_i` for :math:`S - \hat{\Psi}` and :math:`1 - \hat{\psi}_i` for
        :math:`R - \hat{\Psi}`. The diagonal of :math:`S` or :math:`R` is replaced by their respective
        communalities in :math:`\hat{\psi}_i` which gives us the following forms:

        .. math::

            S - \hat{\Psi} =
            \begin{bmatrix}
              \hat{h}^2_1 & s_{12} & \cdots & s_{1p} \\
              s_{21} & \hat{h}^2_2 & \cdots & s_{2p} \\
              \vdots & \vdots & & \vdots \\
              s_{p1} & s_{p2} & \cdots & \hat{h}^2_p \\
            \end{bmatrix}

            R - \hat{\Psi} =
            \begin{bmatrix}
              \hat{h}^2_1 & r_{12} & \cdots & r_{1p} \\
              r_{21} & \hat{h}^2_2 & \cdots & r_{2p} \\
              \vdots & \vdots & & \vdots \\
              r_{p1} & r_{p2} & \cdots & \hat{h}^2_p \\
            \end{bmatrix}

        An initial estimate of the communalities is made using the squared multiple correlation between
        the observation vector :math:`y_i` and the other :math:`p - 1` variables. The squared multiple
        correlation in the case of :math:`R - \hat{\Psi}` is equivalent to the following:

        .. math::

            \hat{h}^2_i = 1 - \frac{1}{r^{ii}}

        Where :math:`r^{ii}` is the :math:`i`th diagonal element of :math:`R^{-1}`. In the case of
        :math:`S - \hat{\Psi}`, the above is multiplied by the variance of the respective variable.

        The factor loadings are then calculated by finding the eigenvalues and eigenvectors of the
        :math:`R - \hat{\Psi}` or :math:`S - \hat{\Psi}` matrix.

        References
        ----------
        Rencher, A. (2002). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        if self.covar is True:
            s = pearson(self.x)
        else:
            s = covar(self.x)

        smc = (1 - 1 / np.diag(np.linalg.inv(s)))

        np.fill_diagonal(s, smc)

        eigvals, loadings, h2, u2, com = self._compute_factors(s)

        proportion_loadings, var_proportion, exp_proportion = self._compute_proportions(loadings, eigvals)

        return loadings, h2, u2, com, proportion_loadings, var_proportion, exp_proportion

    def _iterated_principal_factor(self):
        r"""
        Performs factor analysis using the iterated principal factor method.

        Returns
        -------
        namedtuple
            The factor analysis results are collected into a namedtuple with the following values:
            Factor Loadings
            Communality
            Specific Variance
            Complexity
            Proportion of Loadings
            Proportion of Variance
            Proportion of Variance Explained
            Number of Iterations

        Notes
        -----
        The iterated principal factor method is an extension of the principal factor method that seeks
        improved estimates of the communality. As in the principal factor method, initial estimates of
        :math:`R - \hat{\Psi}` or :math:`S - \hat{\Psi}` are found to obtain :math:`\hat{\Lambda}` from
        which the factors are computed. In the iterated principal factor method, the initial estimates
        of the communality are used to find new communality estimates from the loadings in
        :math:`\hat{\Lambda}` with the following:

        .. math::

            \hat{h}^2_i = \sum^m_{j=1} \hat{\lambda}^2_{ij}

        The values of :math:`\hat{h}^2_i` are then substituted into the diagonal of :math:`R - \hat{\Psi}`
        or :math:`S - \hat{\Psi}` and a new value of :math:`\hat{\Lambda}` is found. This iteration
        continues until the communality estimates converge, though sometimes convergence does not occur.
        Once the estimates converge, the eigenvalues and eigenvectors are calculated from the iterated
        :math:`R - \hat{\Psi}` or :math:`S - \hat{\Psi}` matrix to arrive at the factor loadings.

        References
        ----------
        Rencher, A. (2002). Methods of Multivariate Analysis (2nd ed.).
            Brigham Young University: John Wiley & Sons, Inc.

        """
        minerr = 0.001
        iterations = []

        if self.covar is True:
            s = covar(self.x)
        else:
            s = pearson(self.x)

        smc = (1 - 1 / np.diag(np.linalg.inv(s)))

        np.fill_diagonal(s, smc)

        h2 = np.trace(s)
        err = h2

        while err > minerr:
            eigval, eigvec = np.linalg.eig(s)

            c = eigvec[:, :self.factors]
            d = np.diag(eigval[:self.factors])

            loadings = np.dot(c, np.sqrt(d))

            psi = np.dot(loadings, loadings.T)

            h2_new = np.trace(psi)
            err = np.absolute(h2 - h2_new)
            h2 = h2_new

            iterations.append(h2_new)

            np.fill_diagonal(s, np.diag(psi))

        h2 = np.sum(loadings ** 2, axis=1)

        u2 = 1 - h2

        com = h2 ** 2 / np.sum(loadings ** 4, axis=1)

        proportion_loadings = np.sum(loadings ** 2, axis=0)

        var_proportion, exp_proportion = [], []

        for i in proportion_loadings:
            var_proportion.append(i / np.sum(eigval))
            exp_proportion.append(i / np.sum(proportion_loadings))

        return loadings, h2, u2, com, proportion_loadings, var_proportion, exp_proportion, iterations

    def _compute_factors(self, s):
        eigval, eigvec = np.linalg.eig(s)

        c = eigvec[:, 0:self.factors]
        d = np.diag(eigval[0:self.factors])

        loadings = np.dot(c, np.sqrt(d))

        h2 = np.sum(loadings ** 2, axis=1)

        u2 = np.diag(s) - h2

        com = h2 ** 2 / np.sum(loadings ** 4, axis=1)

        return eigval, loadings, h2, u2, com

    @staticmethod
    def _compute_proportions(loadings, eigvals):
        var_proportion, exp_proportion = [], []
        proportion_loadings = np.sum(loadings ** 2, axis=0)

        for i in proportion_loadings:
            var_proportion.append(i / np.sum(eigvals))
            exp_proportion.append(i / np.sum(proportion_loadings))

        return proportion_loadings, var_proportion, exp_proportion
