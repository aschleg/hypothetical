## Version 0.3.2

* Cochran's Q Test
  * Multiple comparisons are now available by specifying `posthoc=True` when performing the Cochran Q test. 
  * The test now checks that all passed sample observation vectors are the same length before proceeding to the test.

## Version 0.3.1

This release is a quick fix for McNemar's test in the `contingency` module. There was some misleading literature in one 
of the references which led to some incorrect changes in the previous release. This should be resolved now and an 
update to the tests is coming up soon as well.

## Version 0.3.0

### New Additions:

#### Factor Analysis

* Factor Analysis
    * Several algorithms for performing Factor Analysis are available, including principal components, principal 
      factors, and iterated principal factors.

#### Nonparametric

  * Wald-Wolfowitz Two Sample Nonparametric Runs Test
    * A nonparametric test for determining if two independent samples have been drawn from the same population or 
      that they differ in any respect.

#### Analysis of Variance

  * Tests for Homogenity of Variance
    * Bartlett's Test for Homogenity of Variances
    * Levene's Test for Homogenity of Variances
  * Analysis of Variance 
    * Van Der Waerden's (normal scores) Test
  
#### Critical Values

  * Critical Value Tables and Lookup Functions
  * D critical value (used in the Kolomogorov-Smirnov Goodness-of-Fit test) table and lookup function have been added.
      
### Updates and changes:

#### Contingency

  * Chi Square Test of Dependence
    * Updates to methods in `ChiSquareContingency`.
      - The previous measures of association method has been removed in favor 
        of specific methods for each measure of association. These include:
          * Cramer's $V$
          * Phi Coefficient, $\phi$
          * Contingency Coefficient, $C$
          * Tschuprow's Coefficient, $T$ (new)
      - The `test_summary` attribute of an initialized `ChiSquareContingency` class 
        now has separate key-value pairs for each computed association measure.
  * McNemar's Test 
    * The test has been fixed to use the correct cells in the 2x2 contingency table.
    * Continuity correction is now applied by default when performing the test. Setting the parameter `continuity=False` 
      will perform the non-continuity corrected version of the test.
      
#### Nonparametric

  * Median Test 
    * A multiple comparisons posthoc test is now available.
    
* Many updates to documentation and docstrings.

## Version 0.2.1

* Many new statistical and hypothesis testing functions have been added and a ton of refactoring has been performed
  in an effort to create a cohesive library structure that can be utilized as the library grows.
* Below is a list of functions that have been added in the new release:
    * Contingency Tables and Related Tests:
        * Chi-square contingency table test of independence
        * Fisher's Exact Test
        * Cochran's Q test
        * Contingency table marginal sums
        * Contingency table expected frequencies
    * Critical Value Tables and Lookup Functions:
        * Chi-Square Critical Values
        * U-Statistic Critical Values
        * W-Statistic Critical Values
    * Hypothesis Tests:
        * Binomial Test
        * Chi-square one-sample goodness-of-fit
    * Nonparametric Tests:
        * Kruskal-Wallis
        * Mann-Whitney U-test for two independent samples
        * Mood's Median Test
        * Sign test for two related samples
        * Wilcoxon Rank Sum Test
    * Post-Hoc Analysis:
        * Games-Howell Test
        * Tukey's Honestly Significant Difference (HSD)
    * Summary Statistics:
        * Pearson correlation
        * Spearman correlation
        * Covariance (several algorithms available)
        * Variance (several algorithms available)
        * Standard Deviation
        * Variance Condition

# Version 0.1.0

* Initial release version.