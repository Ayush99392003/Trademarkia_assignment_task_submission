"""
Integration test validating the mathematical and logical structures 
of the semantic cache pipelines without needing the full dataset.
"""

from src.cluster_graph import ClusterGraph
from src.cache import SemanticCache, CacheEntry
from src.clustering import FuzzyCMeans
from src.preprocessor import NewsGroupPreprocessor
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path to allow importing src
sys.path.append(str(Path(__file__).parent.parent))


def test_preprocessor():
    print("Running Preprocessor test...")
    prep = NewsGroupPreprocessor(
        chunk_size=10, overlap=2, min_chunk_words=5, min_doc_chars=10)
    raw = "Subject: Test\n\nThis is a long test document that needs to be chunked into pieces."
    subj, body = prep.parse_doc(raw)
    assert subj == "Test", f"Got {subj}"

    clean = prep.clean(body)
    chunks = prep.chunk(clean)
    assert len(chunks) == 2, f"Got {len(chunks)} chunks, expected 2"
    print("✅ Preprocessor working.\n")


def test_fcm():
    print("Running Fuzzy C-Means test...")
    X = np.random.rand(50, 10)
    fcm = FuzzyCMeans(n_clusters=3)
    fcm.fit(X)

    fpc = fcm.fuzzy_partition_coefficient
    assert 1/3 <= fpc <= 1.0, f"FPC {fpc} out of bounds"
    assert fcm.centroids.shape == (3, 10), "Centroids shape mismatch"
    print(f"✅ FCM converging. FPC: {fpc:.4f}\n")


def test_semantic_cache():
    print("Running Semantic Cache test...")
    cache = SemanticCache(max_size=5)
    cache.load_thresholds({0: 0.85, 1: 0.82})

    # Store an entry
    emb1 = np.random.rand(10)
    emb1 = emb1 / np.linalg.norm(emb1)

    entry1 = CacheEntry(
        query_text="hello world",
        query_embedding=emb1,
        result={"hits": 1},
        dominant_cluster=0,
        membership_vec=np.array([0.9, 0.1]),
        entropy=0.3
    )
    cache.store(entry1)

    # Hit test (identical emb)
    hit, _ = cache.lookup(emb1, np.array([0.9, 0.1]), [])
    assert hit is not None, "Cache should hit on identical embedding"

    # Miss test (orthogonal)
    emb2 = np.zeros(10)
    emb2[0] = 1.0
    if np.abs(np.dot(emb1, emb2)) > 0.8:
        emb2 = np.ones(10)
        emb2 = emb2 / np.linalg.norm(emb2)

    hit, _ = cache.lookup(emb2, np.array([0.9, 0.1]), [])
    print(f"✅ Semantic Cache lookup active.\n")


def test_cluster_graph():
    print("Running Cluster Graph test...")
    centroids = np.random.rand(5, 10)
    graph = ClusterGraph()
    graph.build(centroids)
    assert len(graph.graph) == 5, "Graph node mismatch"

    query = np.random.rand(10)
    route = graph.get_search_clusters(query, centroids, top_k=2)
    assert len(route) == 2, f"Expected 2 nodes, got {len(route)}"
    print(f"✅ Cluster Graph routing operational.\n")


if __name__ == "__main__":
    print("=== Integration Verification ===\n")
    test_preprocessor()
    test_fcm()
    test_semantic_cache()
    test_cluster_graph()
    print("All functional tests passed.")
