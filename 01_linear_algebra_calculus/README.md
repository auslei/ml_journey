# 01 Linear algebra and calculus

The maths every later section leans on. Vectors and matrices describe data and models. Decompositions show where the structure is. Gradients say how to improve a model, and autodiff computes them for you.

## Concept map

```
vectors, dot product, norms
        |
matrices as transformations --> determinant, inverse, rank
        |
eigenvalues, SVD  --> PCA (section 04), least squares
        |
derivatives, partial derivatives, gradient --> chain rule, Jacobian
        |
autodiff (reverse mode) --> backpropagation (section 05)
```

## Reading order

1. `01_vectors_and_matrices.ipynb`: the data matrix, `y = Xw`, what a matrix does to space.
2. `02_decompositions.ipynb`: eigenvectors, SVD, low-rank approximation, the normal equation.
3. `03_calculus_and_gradients.ipynb`: gradients, chain rule, Jacobian, gradient descent on a 2D surface.
4. `04_autodiff.ipynb`: a small reverse-mode autodiff engine, checked against numerical gradients.
5. `intuition_test.ipynb`: predict first, then run.

All notebooks run on the core dependencies. PyTorch in notebook 04 is an optional guarded cell.

## What you should be able to explain at the end

- Why a dataset is a matrix and a linear model is one matrix-vector product.
- What the columns of a matrix mean, and what a zero determinant says about information loss.
- What an eigenvector is, how SVD relates to eigendecomposition, and why a few singular values often carry most of the signal.
- Why the normal equation can fail on collinear features and what to use instead.
- Why the gradient points uphill, and what sets a stable learning rate.
- How the chain rule turns into backpropagation, and why reverse mode costs one backward pass for any number of parameters.
- How to check an analytic gradient against a finite difference, and what step size to use.
