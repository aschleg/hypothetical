# hypothetical - Hypothesis and Statistical Testing in Python

[![Build Status](https://travis-ci.org/aschleg/hypothetical.svg?branch=master)](https://travis-ci.org/aschleg/hypothetical)
[![Build status](https://ci.appveyor.com/api/projects/status/i1i1blt9ny3tyi6a?svg=true)](https://ci.appveyor.com/project/aschleg/hypy)
[![Coverage Status](https://coveralls.io/repos/github/aschleg/hypothetical/badge.svg?branch=master)](https://coveralls.io/github/aschleg/hypothetical?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3ceba919fdb34d45af43c044a761ddb8)](https://www.codacy.com/app/aschleg/hypothetical?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=aschleg/hypothetical&amp;utm_campaign=Badge_Grade)
![Python versions](https://img.shields.io/badge/python-3.5%2C%203.6-blue.svg)

Python library for conducting hypothesis and other group comparison tests.

## Available Methods

### Analysis of Variance

* One-way Analysis of Variance (ANOVA)
* One-way Multivariate Analysis of Variance (MANOVA)

### Contingency Tables and Related Tests

* Chi-square test of independence
* Fisher's Exact Test
* McNemar's Test of paired nominal data
* Cochran's Q test

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

### Critical Value Tables and Lookup Functions

* Chi-square
* Wilcoxon Rank Sum W-statistic
* Mann-Whitney U-statistic

### Hypothesis Testing

* Binomial Test
* Chi-square one-sample goodness-of-fit
* t-test
  - paired, one and two sample testing

### Nonparametric Methods

* Kruskal-Wallis (nonparametric equivalent of one-way ANOVA)
* Mann-Whitney (two sample nonparametric variant of t-test)
* Mood's Median test
* Sign test of consistent differences between observation pairs
* Wilcoxon Rank Sum Test (one sample nonparametric variant of paired and one-sample t-test)

### Post-Hoc Analysis

* Tukey's Honestly Significant Difference (HSD)
* Games-Howell (nonparametric)

### Helpful Functions

* Tie Correction for ranked variables
* Contingency table marginal sums
* Contingency table expected frequencies

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

## Installation

## License

MIT