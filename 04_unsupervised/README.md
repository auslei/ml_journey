# 04 Unsupervised learning

Finding structure in data that has no labels. The section covers two jobs: grouping similar points (clustering) and describing the data with fewer numbers (dimensionality reduction). Both reuse ideas from earlier sections: squared-distance cost functions, maximum likelihood, eigendecomposition and the SVD.

## Concept map

```
clustering                                   dimensionality reduction
----------                                   ------------------------
k-means: inertia, Lloyd, k-means++            PCA: variance, eigenvectors, SVD
   |                                              |
hierarchical + DBSCAN: shapes, noise          t-SNE and UMAP: neighbourhoods, nonlinear
   |                                              |
Gaussian mixtures: soft labels, EM, BIC           |
   |                                              |
   +----------------- evaluation -----------------+
        internal metrics, ARI, stability, effect on a supervised model
```

## Reading order

1. `01_kmeans.ipynb`: inertia, Lloyd's algorithm from scratch, k-means++, choosing k (elbow, silhouette), scaling, and what breaks (non-round clusters, unequal sizes, outliers).
2. `02_hierarchical_and_dbscan.ipynb`: linkage and dendrograms, DBSCAN from scratch, `eps` and `min_samples`, the k-distance plot, moons and circles.
3. `03_gaussian_mixtures.ipynb`: EM from scratch in 1D and 2D, covariance types, BIC, k-means as a special case, singular covariances.
4. `04_pca.ipynb`: variance maximisation, PCA through the SVD, explained variance, reconstruction error, whitening, compressing digits, scale and nonlinear data.
5. `05_tsne_and_umap.ipynb`: neighbour probabilities and the KL cost, a small t-SNE from scratch, perplexity, UMAP's `n_neighbors` and `min_dist`, how the plots mislead.
6. `06_evaluating_unsupervised.ipynb`: silhouette, Davies-Bouldin, ARI, bootstrap stability, clusters and PCA as pipeline steps.
7. `intuition_test.ipynb`: predict first, then run.

Data comes from `common/datasets.py`. Digits, wine and iris ship inside scikit-learn. Blobs, moons, circles and the customer table are generated from a seed. The customer table is synthetic: four segments with unequal sizes and features on very different scales. The t-SNE and UMAP notebook subsamples digits to 1,000 rows. It is the slowest notebook, at about 30 seconds, mostly UMAP and repeated t-SNE fits. UMAP compiles with numba on first use, so the first call is slow.

## What you should be able to explain at the end

- What k-means minimises, why each step of Lloyd's algorithm cannot raise it, and why it only finds a local minimum.
- Why inertia cannot choose k, and how the elbow, silhouette and stability each help.
- Why unscaled features and outliers wreck k-means, and why it cuts moons and rings in half.
- How single, complete, average and Ward linkage differ, and how to read a dendrogram.
- What a DBSCAN core, border and noise point are, how to pick `eps` from a k-distance plot, and why varying density defeats it.
- How a Gaussian mixture differs from k-means (soft labels, tilted ellipses, a likelihood) and why k-means is the limit of a GMM with tight round clusters.
- What the E and M steps do, and why BIC can choose the number of components when inertia cannot.
- Why PCA directions are eigenvectors of the covariance matrix, how the SVD gives them, and what the singular values mean.
- How to choose the number of components, and what reconstruction error equals.
- Why PCA fails on unscaled data, ring-shaped data, and signal that lives in low-variance directions.
- What the t-SNE cost rewards, what perplexity does, and why cluster sizes, distances between clusters and layout are not evidence.
- How UMAP's `n_neighbors` and `min_dist` change the picture.
- How to judge a clustering without labels, and why a supervised score is the test that matters when the result feeds a model.
