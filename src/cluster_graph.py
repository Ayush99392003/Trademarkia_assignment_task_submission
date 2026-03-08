"""
Semantic Cluster-to-Cluster routing graph based on centroid cosine similarities.
Allows adaptive semantic expansion bounds at search time.
"""

from typing import Dict, List, Tuple
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ClusterGraph:
    """
    Map of connected clusters based on fuzzy centroid proximity.
    Used during search cache logic to select secondary buckets dynamically.
    """

    def __init__(self):
        self.graph: Dict[int, List[Tuple[int, float]]] = {}
        self.edge_threshold: float = 0.0

    def build(self, centroids: np.ndarray, threshold: float = None):
        """
        Creates weighted edges mapping clusters that share borders.
        """
        K = len(centroids)
        if K <= 1:
            return

        sim_matrix = cosine_similarity(centroids)

        # Apply automatic threshold constraint if explicit param isn't manually given
        if threshold is None:
            # We want off-diagonal elements above the average pair threshold
            all_sims = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            self.edge_threshold = float(all_sims.mean() + 0.5 * all_sims.std())
        else:
            self.edge_threshold = float(threshold)

        for i in range(K):
            neighbours = []
            for j in range(K):
                if i != j and sim_matrix[i, j] > self.edge_threshold:
                    neighbours.append((j, float(sim_matrix[i, j])))

            # Descending similarity sort overrides typical ID order
            self.graph[i] = sorted(neighbours, key=lambda x: -x[1])

    def get_search_clusters(self, query_embedding: np.ndarray,
                            centroids: np.ndarray, top_k: int = 3) -> List[int]:
        """
        Derives an entry cluster, then branches to adjacent logical concepts.
        """
        if not self.graph:
            return []

        # Find absolute best match entry point to begin traversal
        sims = cosine_similarity(query_embedding.reshape(1, -1), centroids)[0]
        entry = int(np.argmax(sims))

        # Branch to pre-computed linked categories that belong to graph neighbourhood
        neighbours = [cid for cid, w in self.graph.get(entry, [])[:top_k - 1]]

        return [entry] + neighbours

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({
                "graph": self.graph,
                "edge_threshold": self.edge_threshold
            }, f)

    def load(self, path: Path):
        if not path.exists():
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data["graph"]
            self.edge_threshold = data["edge_threshold"]
