"""
Semantic cache layer designed around Fuzzy concept routing and
adaptive density thresholds. Replaces generic instances with
logic specifically suited for overlapping embeddings.
"""

import time
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy


@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray
    result: dict
    dominant_cluster: int
    membership_vec: np.ndarray
    entropy: float
    hit_count: int = 0
    timestamp: float = 0.0


class SemanticCache:
    """
    Cluster-segregated semantic memory. Employs entropy routing
    and variance-based thresholds.
    """

    def __init__(self, max_size: int = 1000):
        self.buckets: Dict[int, List[CacheEntry]] = defaultdict(list)
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        self.adaptive_thresholds: Dict[int, float] = {}

    def load_thresholds(self, thresholds: Dict[int, float]):
        """
        Receives external per-cluster base threshold calibrations
        computed during FCM.
        """
        self.adaptive_thresholds = thresholds

    def _get_theta(self, cluster_id: int) -> float:
        # Fallback if standard 0.82 wasn't supplied by config properly
        return self.adaptive_thresholds.get(cluster_id, 0.82)

    def _check_bucket(self, query_embedding: np.ndarray,
                      bucket_id: int) -> Optional[CacheEntry]:
        """
        Scan a specific cluster bucket for a semantic match
        above its learned theta.
        """
        bucket = self.buckets.get(bucket_id, [])
        if not bucket:
            return None

        # Build matrix
        cached_embs = np.vstack([entry.query_embedding for entry in bucket])

        # Calculate cosine similarity (1, N)
        sims = cosine_similarity(
            query_embedding.reshape(1, -1), cached_embs)[0]

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        theta = self._get_theta(bucket_id)

        if best_sim >= theta:
            return bucket[best_idx]

        return None

    def lookup(self, query_embedding: np.ndarray,
               membership_vec: np.ndarray,
               graph_neighbours: List[int]) -> Tuple[
                   Optional[CacheEntry], List[int]]:
        """
        Three stage lookup using entropy routing.
        """
        n_clusters = len(membership_vec)
        if n_clusters == 0:
            return None, []

        membership_entropy = float(
            entropy(np.clip(membership_vec, 1e-10, 1.0)))
        max_entropy = np.log(n_clusters)

        # 0 = absolute certainty, 1 = maximum ambiguity
        normalised_entropy = (membership_entropy / max_entropy
                              if max_entropy > 0 else 0)

        # If highly uncertain, boundary routing looks at top 2
        if normalised_entropy > 0.8 and n_clusters >= 2:
            top_clusters = np.argsort(membership_vec)[-2:][::-1].tolist()
        else:
            top_clusters = [int(np.argmax(membership_vec))]

        # Remember which ones we checked natively
        # to guide Chroma search on MISS
        checked_clusters = set()

        # Stage 1: Primary bucket check (Entropy routed)
        for cid in top_clusters:
            checked_clusters.add(cid)
            hit = self._check_bucket(query_embedding, cid)
            if hit:
                self.hit_count += 1
                self._record_access(hit)
                return hit, list(checked_clusters)

        # Stage 2: Graph neighbour check
        # Only traverse neighbours not already checked by entropy routing
        for nid in graph_neighbours:
            if nid in checked_clusters:
                continue
            checked_clusters.add(nid)
            hit = self._check_bucket(query_embedding, nid)
            if hit:
                self.hit_count += 1
                self._record_access(hit)
                return hit, list(checked_clusters)

        # Stage 3: Global fallback (only triggered if cache is mostly empty)
        total_cache_size = sum(len(b) for b in self.buckets.values())
        if total_cache_size > 0 and total_cache_size < 20:
            for cid in self.buckets.keys():
                if cid in checked_clusters:
                    continue
                checked_clusters.add(cid)
                hit = self._check_bucket(query_embedding, cid)
                if hit:
                    self.hit_count += 1
                    self._record_access(hit)
                    return hit, list(checked_clusters)

        self.miss_count += 1
        return None, list(checked_clusters)

    def store(self, entry: CacheEntry):
        """
        Caches a new result, evicting the oldest element if full.
        """
        self._evict_if_full()
        entry.timestamp = time.time()
        self.buckets[entry.dominant_cluster].append(entry)

    def _record_access(self, entry: CacheEntry):
        """Update LFU/LRU metrics on hit."""
        entry.hit_count += 1
        entry.timestamp = time.time()

    def _evict_if_full(self):
        """
        Hybrid ClusterLFU / LRU eviction.
        Target least recently used entry overall bounding maximum capacity.
        """
        total = sum(len(b) for b in self.buckets.values())
        if total < self.max_size:
            return

        oldest_cluster, oldest_idx = None, None
        oldest_time = float('inf')

        for cid, entries in self.buckets.items():
            for i, entry in enumerate(entries):
                # Simple LRU across all buckets
                if entry.timestamp < oldest_time:
                    oldest_time = entry.timestamp
                    oldest_cluster, oldest_idx = cid, i

        if oldest_cluster is not None:
            self.buckets[oldest_cluster].pop(oldest_idx)

    def stats(self) -> dict:
        """Dashboard overview metric gathering."""
        total = sum(len(b) for b in self.buckets.values())
        requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / requests) if requests > 0 else 0.0

        cluster_counts = {cid: len(b) for cid, b in self.buckets.items()}

        return {
            "total_entries": total,
            "max_size": self.max_size,
            "requests": requests,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "cluster_distribution": cluster_counts
        }

    def flush(self):
        self.buckets.clear()
        self.hit_count = 0
        self.miss_count = 0
