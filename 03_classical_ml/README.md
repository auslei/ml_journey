# 03 Classical machine learning

The models that still do most of the practical work in tabular data, and the habits (splitting, validation, regularisation, metrics) that every later section reuses. The section runs in two parts. Part A covers the workflow and regression. Part B covers classification, trees, ensembles, metrics and a capstone.

## Concept map

```
workflow: train/validation/test, cross-validation, leakage, pipelines
        |
linear regression: MSE, normal equation, gradient descent, scaling
        |
feature engineering, polynomial features --> overfitting
        |
gradient descent variants: batch, SGD, mini-batch, learning rate, momentum
        |
bias-variance --> Ridge (L2), Lasso (L1), ElasticNet, tuning alpha by CV
        |
(Part B) logistic regression --> kNN --> SVM --> trees --> ensembles
        |
(Part B) classification metrics --> scorecard capstone
```

## Reading order

Part A (done):

1. `00_ml_workflow.ipynb`: splits, cross-validation, the cost function idea, overfitting, data leakage and pipelines.
2. `10_linear_regression.ipynb`: MSE, the normal equation, gradient descent from scratch, scaling, coefficients, collinearity, outliers.
3. `11_polynomial_and_features.ipynb`: feature engineering, polynomial features, under and overfitting on a controlled curve.
4. `12_gradient_descent_variants.ipynb`: batch, SGD and mini-batch from scratch, learning rate, momentum, early stopping.
5. `13_bias_variance_regularisation.ipynb`: the bias-variance decomposition by simulation, Ridge, Lasso, ElasticNet, regularisation paths, tuning alpha.
6. `regression_intuition_test.ipynb`: predict first, then run.

Part B (done):

7. `20_logistic_regression.ipynb`: sigmoid, log-loss from Bernoulli maximum likelihood, gradient descent from scratch, regularisation with C, softmax, separable data and imbalance.
8. `21_knn.ipynb`: distances, choice of k, scaling, the curse of dimensionality, digits.
9. `22_svm.ipynb`: max margin, hinge loss, soft margin, RBF kernel, tuning C and gamma.
10. `23_decision_trees.ipynb`: Gini and entropy by hand, depth and pruning, feature importance, instability.
11. `24_ensembles.ipynb`: bagging, random forest, boosting from scratch, tuning, comparison table.
12. `25_classification_metrics.ipynb`: confusion matrix, precision and recall, ROC and PR curves, threshold choice from costs, calibration.
13. `26_scorecard_capstone.ipynb`: binning, weight of evidence, logistic regression, Gini and KS, points to double the odds.
14. `intuition_test.ipynb`: the closing test for Part B. Part A's questions live in `regression_intuition_test.ipynb`.

Every notebook runs on the core dependencies. Data comes from `common/datasets.py`: California housing is downloaded once into the gitignored `data/` folder. Iris, breast cancer and digits ship inside scikit-learn. The noisy curve, sparse-regression and credit-default sets are generated from a seed. The credit data are synthetic because no real credit dataset downloads without an account.

## What you should be able to explain at the end of Part A

- Why training error always falls with model flexibility, and why validation error does not.
- Why a pipeline stops leakage, and what a suspiciously good score usually means.
- How the normal equation and gradient descent reach the same weights, and when each one is the better tool.
- Why unscaled features make gradient descent crawl or diverge, with the condition number as the evidence.
- What a polynomial regression is (linear in the weights, not the inputs), and why high degrees extrapolate badly.
- How batch, SGD and mini-batch differ, what a learning rate that is too high looks like, and what momentum adds.
- How expected error splits into bias squared, variance and noise, and why a penalty trades a little of the first for a lot of the second.
- Why L1 produces exact zeros and L2 only shrinks, and how both connect to the priors in `02_probability/03`.
- Which of Ridge, Lasso and ElasticNet to try for a given problem, and how to tune alpha.

## What you should be able to explain at the end of Part B

- Where log-loss comes from (Bernoulli maximum likelihood), how to read a logistic coefficient as an odds multiplier, and why separable data sends the weights to infinity.
- How kNN, SVM and trees each draw a boundary, and which knob (k, C and gamma, depth) moves a model along the bias-variance curve.
- Which models need scaling (kNN, SVM, logistic with a penalty) and which do not (trees).
- Why the curse of dimensionality hurts distance-based models.
- Why bagging cuts variance and boosting cuts bias, and what a random forest adds to bagging.
- Why accuracy misleads on imbalanced data, how precision, recall, ROC and PR curves differ, and how to pick a threshold from costs.
- What weight of evidence, Gini, KS and points to double the odds mean in a credit scorecard.
