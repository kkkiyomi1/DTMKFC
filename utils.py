"""
utils.py

Utility functions for graph construction, constraint generation, and clustering evaluation.
"""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from sklearn.preprocessing import normalize
from collections import Counter
from scipy.optimize import linear_sum_assignment


def rbf_affinity(X, gamma=0.5):
    """Compute full RBF similarity matrix."""
    sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) +                np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
    A = np.exp(-gamma * sq_dists)
    np.fill_diagonal(A, 0)
    return A


def kNN_affinity(X, k=10, metric='cosine'):
    """Build a k-NN graph using cosine or Euclidean similarity."""
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    if metric == 'cosine':
        sim = cosine_similarity(X)
    else:
        dist = euclidean_distances(X)
        sim = -dist
    A = np.zeros_like(sim)
    for i in range(sim.shape[0]):
        knn_idx = np.argsort(sim[i])[-k - 1:-1]
        A[i, knn_idx] = sim[i, knn_idx]
    A = np.maximum(A, A.T)
    return A


def sample_constraints(labels, num_constraints=100, ratio=0.5, seed=42):
    """Generate must-link and cannot-link constraints from ground truth."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_pairs)

    must_link, cannot_link = [], []
    for i, j in all_pairs:
        if labels[i] == labels[j] and len(must_link) < int(num_constraints * ratio):
            must_link.append((i, j))
        elif labels[i] != labels[j] and len(cannot_link) < int(num_constraints * (1 - ratio)):
            cannot_link.append((i, j))
        if len(must_link) + len(cannot_link) >= num_constraints:
            break
    return must_link, cannot_link


def clustering_accuracy(y_true, y_pred):
    """Hungarian-matched clustering accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) / y_pred.size


def purity_score(y_true, y_pred):
    """Purity = fraction of dominant label in each predicted cluster."""
    y_voted_labels = np.zeros_like(y_pred)
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        true_labels = y_true[mask]
        if len(true_labels) == 0:
            continue
        majority = Counter(true_labels).most_common(1)[0][0]
        y_voted_labels[mask] = majority
    return accuracy_score(y_true, y_voted_labels)


def evaluate_clustering(y_true, y_pred):
    """Print ACC / NMI / Purity."""
    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    print(f'[INFO] ACC    = {acc:.4f}')
    print(f'[INFO] NMI    = {nmi:.4f}')
    print(f'[INFO] Purity = {pur:.4f}')
    return acc, nmi, pur
