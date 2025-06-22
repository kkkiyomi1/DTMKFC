"""
dtmkfc.py

Core implementation of the DTMKFC algorithm:
Dynamic Topology-aware Multi-view Knowledge-guided Semi-supervised Clustering.
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh


class DTMKFC:
    def __init__(self, n_clusters, n_neighbors=10, gamma=0.5, n_iter=10, alpha=1.0, beta=0.1):
        """
        Parameters:
            n_clusters (int): Number of output clusters.
            n_neighbors (int): k for kNN graph.
            gamma (float): Kernel width for RBF.
            n_iter (int): Iterations of graph refinement.
            alpha (float): Weight for constraint integration.
            beta (float): Weight for dynamic topology refinement.
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.labels_ = None

    def _construct_affinity(self, X):
        """Construct initial RBF-based similarity matrix."""
        K = rbf_kernel(X, gamma=self.gamma)
        np.fill_diagonal(K, 0)
        return K

    def _spectral_clustering(self, W):
        """Apply spectral clustering to affinity matrix."""
        L = laplacian(W, normed=True)
        _, eigvecs = eigh(L, subset_by_index=[0, self.n_clusters - 1])
        normed_eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
        return normed_eigvecs

    def _refine_affinity(self, A, constraints):
        """Refine affinity matrix using constraints."""
        must_link, cannot_link = constraints
        W_new = A.copy()
        for i, j in must_link:
            W_new[i, j] += self.alpha
            W_new[j, i] += self.alpha
        for i, j in cannot_link:
            W_new[i, j] -= self.alpha
            W_new[j, i] -= self.alpha
        W_new = np.clip(W_new, 0, 1)
        return W_new

    def _fuse_views(self, affinity_list):
        """Average fusion across views."""
        return np.mean(np.stack(affinity_list, axis=0), axis=0)

    def fit(self, views, constraints=None):
        """
        Train DTMKFC on multi-view data.

        Parameters:
            views (List[np.ndarray]): List of views (each shape: [n_samples, n_features])
            constraints (Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]):
                must-link and cannot-link constraints
        """
        n_samples = views[0].shape[0]
        assert all(view.shape[0] == n_samples for view in views), "View size mismatch"

        affinity_list = [self._construct_affinity(X) for X in views]
        W = self._fuse_views(affinity_list)

        for it in range(self.n_iter):
            if constraints:
                W = self._refine_affinity(W, constraints)
            Z = self._spectral_clustering(W)
            W = np.dot(Z, Z.T) + self.beta * W
            W = np.clip(W, 0, 1)

        from sklearn.cluster import KMeans
        self.labels_ = KMeans(n_clusters=self.n_clusters).fit_predict(Z)
        return self

    def predict(self):
        return self.labels_
