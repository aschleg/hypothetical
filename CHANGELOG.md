## Version 0.2.2

* New statistical and hypothesis methods have been added, including:
  * Tests for Homogenity of Variance
    * Bartlett's Test for Homogenity of Variances
    * Levene's Test for Homogenity of Variances
  * Analysis of Variance 
    * Van Der Waerden's (normal scores) Test
  * Factor Analysis
    * Several algorithms for performing Factor Analysis are avaiable, including principal components, principal 
      factors, and iterated principal factors.
      
* Fixes and Updates:
  * One-Sample Runs Test
    * The test now excepts arrays with more than two unique values.

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