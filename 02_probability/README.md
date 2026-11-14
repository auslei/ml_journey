# 02 Probability

Models predict distributions, not single answers. A classifier outputs how likely each class is, and treating that output as a real probability is what makes it useful. This section builds the vocabulary of probability, works through Bayes' theorem, and then draws the line from probability to the loss functions used in every later section. The payoff is that squared error, cross-entropy and L2 regularisation stop being formulas to memorise and become consequences of a noise assumption and a prior.

## Concept map

```
random variables, distributions (Bernoulli, binomial, normal, Poisson)
        |
expectation, variance --> law of large numbers, CLT
        |
joint, conditional, independence
        |
Bayes' theorem: prior x likelihood = posterior --> naive Bayes (section 03)
        |
maximum likelihood --> squared error (Gaussian), cross-entropy (Bernoulli, categorical)
        |
MAP --> L2 regularisation (Gaussian prior)
```

## Reading order

1. `01_probability_basics.ipynb`: distributions, expectation, variance, LLN and CLT by simulation, conditional probability, independence.
2. `02_bayes.ipynb`: the base-rate trap, the Beta-Binomial update, how data overwhelms a prior, naive Bayes from scratch checked against sklearn.
3. `03_probability_in_ml.ipynb`: where squared error, cross-entropy, softmax and L2 regularisation come from.
4. `intuition_test.ipynb`: predict first, then run.

All notebooks run on the core dependencies (numpy, scipy, pandas, matplotlib, scikit-learn).

## What you should be able to explain at the end

- Why a positive result on a 90% accurate test for a 1% disease means about a 9% chance of being sick, because the healthy majority produces most of the positives.
- What a conjugate prior is, why Beta(a, b) behaves like a + b pseudo-flips, and how enough data overwhelms any prior.
- Why the average of n samples has error of order 1/sqrt(n), and the two cases where that fails (heavy tails with no finite mean, and dependent or duplicated rows).
- Why naive Bayes is naive, and what zero counts and repeated features do to its probabilities.
- How Gaussian noise gives squared error, Bernoulli gives cross-entropy, and a Gaussian prior gives L2, each as a negative log-likelihood.
- Why softmax outputs are a categorical distribution, and why logistic regression diverges on separable data without a penalty.
