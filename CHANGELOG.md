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
  * Wald-Wolfowitz Two Sample Nonparametric Runs Test
    * A nonparametric test for determining if two independent samples have been drawn from the same population or 
      that they differ in any respect.
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