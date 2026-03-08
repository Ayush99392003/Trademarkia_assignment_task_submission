"""
Extracts document embeddings from ChromaDB, evaluates Fuzzy C-Means for a range of K,
computes adaptive clustering heuristics, and saves the topology assets.
"""

from src.cluster_graph import ClusterGraph
from src.clustering import FuzzyCMeans, ClusterEvaluator
from src.vector_store import VectorStore
from src.config import settings
from rich.progress import track
from rich.table import Table
from rich.console import Console
import numpy as np
import sys
import pickle
from pathlib import Path

# Add project root to sys.path to allow importing src
sys.path.append(str(Path(__file__).parent.parent))


def main():
    console = Console()
    console.print("[bold cyan]Starting Fuzzy Clustering Pipeline[/bold cyan]")

    # 1. Connect to VectorStore and retrieve doc embeddings
    vector_store = VectorStore(
        db_dir=settings.db_dir,
        space=settings.hnsw_space,
        ef_construction=settings.hnsw_ef_construction,
        m=settings.hnsw_m
    )

    doc_ids, X = vector_store.get_all_doc_embeddings()
    n_docs = len(doc_ids)

    if n_docs == 0:
        console.print(
            "[bold red]No documents found in Vector Store. Run ingest first![/bold red]")
        sys.exit(1)

    console.print(f"Loaded {n_docs} document embeddings for clustering.")

    # 2. Evaluate K range
    k_range = settings.k_range
    results = []

    best_k = None
    best_score = -1
    best_model = None

    # Use Rich track for progress indication over K
    for k in track(k_range, description="Evaluating K candidates..."):
        fcm = FuzzyCMeans(n_clusters=k, m=settings.fcm_m_fuzziness,
                          max_iter=settings.fcm_max_iter, tol=settings.fcm_tol)
        fcm.fit(X)

        fpc = fcm.fuzzy_partition_coefficient

        # Hard assignments for silhouette calc
        hard_labels = np.argmax(fcm.U, axis=1)
        sil_score = ClusterEvaluator.silhouette(X, hard_labels)

        # Simple heuristic: Combine them
        combined_score = fpc + sil_score

        results.append({
            "k": k,
            "fpc": fpc,
            "sil": sil_score,
            "score": combined_score,
            "model": fcm,
            "labels": hard_labels
        })

        if combined_score > best_score:
            best_score = combined_score
            best_k = k
            best_model = fcm

    # 3. Print Selection Table
    table = Table(title="K Selection Evaluation")
    table.add_column("K Clusters", justify="right", style="cyan")
    table.add_column("FPC Score", justify="right", style="green")
    table.add_column("Silhouette", justify="right", style="magenta")
    table.add_column("Combined", justify="right")

    for res in results:
        style = "bold yellow" if res["k"] == best_k else ""
        table.add_row(
            str(res["k"]),
            f"{res['fpc']:.4f}",
            f"{res['sil']:.4f}",
            f"{res['score']:.4f}",
            style=style
        )

    console.print(table)
    console.print(f"Selected Optimal K: [bold yellow]{best_k}[/bold yellow]")

    # 4. Compute Adaptive Thresholds
    console.print("Computing variance-based adaptive thresholds...")
    adaptive_thresholds = {}

    for k in range(best_k):
        # Documents assigned to cluster k
        cluster_docs_idx = np.where(np.argmax(best_model.U, axis=1) == k)[0]

        if len(cluster_docs_idx) < 2:
            adaptive_thresholds[k] = settings.cache_base_theta
            continue

        cluster_embeddings = X[cluster_docs_idx]
        centroid = best_model.centroids[k]

        # Intra-cluster variance calculation
        dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        variance = float(np.std(dists))

        # The logic: map variance deviation to ±0.1 threshold adjustment
        adjustment = (variance - 0.5) * 0.2
        adj_theta = np.clip(settings.cache_base_theta - adjustment, 0.65, 0.95)
        adaptive_thresholds[k] = float(adj_theta)

    # 5. Build Cluster Graph
    console.print("Building Semantic Routing Graph...")
    graph = ClusterGraph()
    graph.build(best_model.centroids)  # Auto-thresholds

    # 6. Save Artifacts
    console.print("Saving analytical artifacts...")
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(settings.artifacts_dir / "fcm_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)

    graph.save(settings.artifacts_dir / "cluster_graph.pkl")

    with open(settings.artifacts_dir / "cluster_thresholds.pkl", 'wb') as f:
        pickle.dump(adaptive_thresholds, f)

    # 7. Update Vector Store Metadata
    console.print("Updating Vector Store cluster mappings...")
    best_labels = np.argmax(best_model.U, axis=1)

    for i in track(range(n_docs), description="Tagging chunks..."):
        doc_id = doc_ids[i]
        cluster_id = int(best_labels[i])
        dominant_membership = float(best_model.U[i, cluster_id])

        vector_store.update_cluster_metadata(
            doc_id, cluster_id, dominant_membership
        )

    console.print("[bold green]Clustering Engine Finalised![/bold green]")


if __name__ == "__main__":
    main()
