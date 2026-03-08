"""
Fuzzy C-Means implementation from scratch for overlapping topic modelling.
Also provides Gaussian Mixture Model (GMM) comparison and clustering metrics.
"""

import numpy as np
from scipy.stats import entropy
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


class FuzzyCMeans:
    """
    Mathematical implementation of Fuzzy C-Means (FCM) allowing soft memberships.
    """

    def __init__(self, n_clusters: int, m: float = 2.0, max_iter: int = 150, tol: float = 1e-4):
        self.K = n_clusters
        self.m = m  # 1 = hard, infinity = uniform
        self.max_iter = max_iter
        self.tol = tol

        self.centroids = None
        self.U = None  # Membership matrix [n_docs x K]

    def fit(self, X: np.ndarray):
        n = X.shape[0]
        # Random initialisation ensuring sum = 1 per row
        self.U = np.random.dirichlet(np.ones(self.K), size=n)

        for iteration in range(self.max_iter):
            U_prev = self.U.copy()

            # Update centroids based on membership degrees
            Um = self.U ** self.m
            self.centroids = (Um.T @ X) / Um.sum(axis=0)[:, None]

            # Update memberships
            distances = self._compute_distances(X)
            self.U = self._update_memberships(distances)

            if np.max(np.abs(self.U - U_prev)) < self.tol:
                break

        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        # Euclidean distance from every point to every centroid
        dist = np.zeros((X.shape[0], self.K))
        for j, c in enumerate(self.centroids):
            # linalg.norm computes distance per row
            dist[:, j] = np.linalg.norm(X - c, axis=1)

        # Clip to prevent divide-by-zero for identical points
        return np.clip(dist, 1e-10, None)

    def _update_memberships(self, distances: np.ndarray) -> np.ndarray:
        exp = 2.0 / (self.m - 1)
        U = np.zeros_like(distances)

        for j in range(self.K):
            # Ratio of distance to cluster j against all other dists
            ratio = distances[:, j:j+1] / distances
            U[:, j] = 1.0 / (ratio ** exp).sum(axis=1)

        return U

    @property
    def fuzzy_partition_coefficient(self) -> float:
        """
        FPC score: 1/K = fully fuzzy (random), 1.0 = fully crisp (like KMeans).
        Higher is better.
        """
        if self.U is None:
            return 0.0
        return float(np.mean(self.U ** 2) * self.K)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Finds memberships for new unrecognised vectors.
        """
        distances = self._compute_distances(X)
        return self._update_memberships(distances)


class ClusterEvaluator:
    """
    Utilities for evaluating cluster qualities and alternative approaches.
    """

    @staticmethod
    def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Geometric separation wrapper. Peak indicates best dense separation.
        """
        if len(np.unique(labels)) <= 1:
            return -1.0
        return float(silhouette_score(X, labels))

    @staticmethod
    def gm_compare(X: np.ndarray, K: int):
        """
        Fits GMM to calculate Bayesian Information Criterion (BIC).
        Valuable to contrast against FCM results.
        """
        gmm = GaussianMixture(n_components=K, covariance_type='full',
                              random_state=42, max_iter=150)
        gmm.fit(X)
        return {
            "bic": float(gmm.bic(X)),
            "soft_memberships": gmm.predict_proba(X)
        }

    @staticmethod
    def membership_entropy(u_row: np.ndarray) -> float:
        """
        High entropy = document is uncertain, lives on boundary.
        Low entropy = distinct assignment to a single cluster.
        """
        # Converts tiny negatives / zeros to very small number to prevent NaN
        u_row = np.clip(u_row, 1e-10, 1.0)
        u_row = u_row / u_row.sum()
        return float(entropy(u_row))
