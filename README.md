# hypothetical - Hypothesis and Statistical Testing in Python

[![Build Status](https://travis-ci.org/aschleg/hypothetical.svg?branch=master)](https://travis-ci.org/aschleg/hypothetical)
[![Build status](https://ci.appveyor.com/api/projects/status/i1i1blt9ny3tyi6a?svg=true)](https://ci.appveyor.com/project/aschleg/hypy)
[![Coverage Status](https://coveralls.io/repos/github/aschleg/hypothetical/badge.svg?branch=master)](https://coveralls.io/github/aschleg/hypothetical?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3ceba919fdb34d45af43c044a761ddb8)](https://www.codacy.com/app/aschleg/hypothetical?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=aschleg/hypothetical&amp;utm_campaign=Badge_Grade)
[![Dependencies](https://img.shields.io/librariesio/github/aschleg/hypothetical.svg?label=dependencies)](https://libraries.io/github/aschleg/hypothetical)
![Python versions](https://img.shields.io/badge/python-3.5%2C%203.6%2C%203.7-blue.svg)

Python library for conducting hypothesis and other group comparison tests.

## Installation

The best way to install `hypothetical` is through `pip`.

```bash
pip install hypothetical
```

For those interested, the most recent development version of the library can also be installed by cloning or 
downloading the repo.

~~~ bash
git clone git@github.com:aschleg/hypothetical.git
cd hypothetical
python setup.py install
~~~

## Available Methods

### Analysis of Variance

* One-way Analysis of Variance (ANOVA)
* One-way Multivariate Analysis of Variance (MANOVA)
* Bartlett's Test for Homogenity of Variances
* Levene's Test for Homogenity of Variances
* Van Der Waerden's (normal scores) Test

### Contingency Tables and Related Tests

* Chi-square test of independence
* Fisher's Exact Test
* McNemar's Test of paired nominal data
* Cochran's Q test
* D critical value (used in the Kolomogorov-Smirnov Goodness-of-Fit test).

### Critical Value Tables and Lookup Functions

* Chi-square statistic
* r (one-sample runs test and Wald-Wolfowitz runs test) statistic 
* Mann-Whitney U-statistic
* Wilcoxon Rank Sum W-statistic

### Descriptive Statistics

* Kurtosis
* Skewness
* Mean Absolute Deviation
* Pearson Correlation
* Spearman Correlation
* Covariance
  - Several algorithms for computing the covariance and covariance matrix of 
    sample data are available
* Variance
  - Several algorithms are also available for computing variance.
* Simulation of Correlation Matrices
  - Multiple simulation algorithms are available for generating correlation matrices.

### Factor Analysis

* Several algorithms for performing Factor Analysis are available, including principal components, principal 
      factors, and iterated principal factors.

### Hypothesis Testing

* Binomial Test
* t-test
  - paired, one and two sample testing

### Nonparametric Methods

* Friedman's test for repeated measures
* Kruskal-Wallis (nonparametric equivalent of one-way ANOVA)
* Mann-Whitney (two sample nonparametric variant of t-test)
* Mood's Median test
* One-sample Runs Test
* Wald-Wolfowitz Two-Sample Runs Test
* Sign test of consistent differences between observation pairs
* Wald-Wolfowitz Two-Sample Runs test
* Wilcoxon Rank Sum Test (one sample nonparametric variant of paired and one-sample t-test)

### Normality and Goodness-of-Fit Tests

* Chi-square one-sample goodness-of-fit
* Jarque-Bera test

### Post-Hoc Analysis

* Tukey's Honestly Significant Difference (HSD)
* Games-Howell (nonparametric)

### Helpful Functions

* Add noise to a correlation or other matrix
* Tie Correction for ranked variables
* Contingency table marginal sums
* Contingency table expected frequencies
* Runs and count of runs

## Goal

The goal of the `hypothetical` library is to help bridge the gap in statistics and hypothesis testing 
capabilities of Python closer to that of R. Python has absolutely come a long way with several popular and 
amazing libraries that contain a myriad of statistics functions and methods, such as [`numpy`](http://www.numpy.org/), 
[`pandas`](https://pandas.pydata.org/), and [`scipy`](https://www.scipy.org/); however, it is my humble opinion that 
there is still more that can be done to make Python an even better language for data and statistics computation. Thus, 
it is my hope with the `hypothetical` library to build on top of the wonderful Python packages listed earlier and 
create an easy-to-use, feature complete, statistics library. At the end of the day, if the library helps a user 
learn more about statistics or get the information they need in an easy way, then I consider that all the success 
I need!

## Requirements

* Python 3.5+
* `numpy>=1.13.0`
* `numpy_indexed>=0.3.5`
* `pandas>=0.22.0`
* `scipy>=1.1.0`
* `statsmodels>=0.9.0`

## License

MIT