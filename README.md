# hypothetical - Hypothesis and Statistical Testing in Python

[![Build Status](https://travis-ci.org/aschleg/hypothetical.svg?branch=master)](https://travis-ci.org/aschleg/hypothetical)
[![Build status](https://ci.appveyor.com/api/projects/status/i1i1blt9ny3tyi6a?svg=true)](https://ci.appveyor.com/project/aschleg/hypy)
[![Coverage Status](https://coveralls.io/repos/github/aschleg/hypothetical/badge.svg?branch=master)](https://coveralls.io/github/aschleg/hypothetical?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3ceba919fdb34d45af43c044a761ddb8)](https://www.codacy.com/app/aschleg/hypothetical?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=aschleg/hypothetical&amp;utm_campaign=Badge_Grade)
![Python versions](https://img.shields.io/badge/python-3.4%2C%203.5%2C%203.6-blue.svg)

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

### Correlation and Covariance

* Pearson Correlation
* Spearman Correlation
* Covariance
  - Several algorithms for computing the covariance and covariance matrix of 
    sample data are available

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

* Mann-Whitney (two sample nonparametric variant of t-test)
* Mood's Median test
* Wilcoxon Rank Sum Test (one sample nonparametric variant of paired and one-sample t-test)
* Kruskal-Wallis (nonparametric equivalent of one-way ANOVA)

### Post-Hoc Analysis

* Tukey's Honestly Significant Difference (HSD)
* Games-Howell (nonparametric)

### Helpful Functions

* Tie Correction for ranked variables
* Contingency table marginal sums
* Contingency table expected frequencies

## Goal

## Motivation

## Requirements

* Python 3.4+
* `pandas >= 0.22.0`
* `numpy >= 1.13.0`
* `numpy_indexed >= 0.3.5`

## Installation

## License

MIT