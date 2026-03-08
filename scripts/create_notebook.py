"""
Generates the analysis.ipynb notebook with pre-filled cells.
"""
import json
from pathlib import Path


def create_cell(cell_type, source):
    if isinstance(source, str):
        source = [line + '\n' for line in source.split('\n')]
        if source and source[-1].endswith('\n'):
            source[-1] = source[-1][:-1]

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main():
    project_root = Path(__file__).parent.parent
    out_dir = project_root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = []

    cells.append(create_cell(
        "markdown", "# Trademarkia Semantic Search - Analysis\nExploring K selection, Fuzzy topologies, and adaptive thresholds."))

    setup_code = '''import sys\nfrom pathlib import Path\nimport pickle\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport umap\n\n# Add project root to sys.path\nproject_root = Path("..").resolve()\nsys.path.append(str(project_root))\n\nfrom src.config import settings\nfrom src.vector_store import VectorStore\n\n# Plotting config\nsns.set_theme(style="whitegrid")\nplt.rcParams['figure.figsize'] = (10, 6)'''
    cells.append(create_cell("code", setup_code))

    load_code = '''# 1. Load Data\nvector_store = VectorStore(\n    db_dir=settings.db_dir,\n    space=settings.hnsw_space,\n    ef_construction=settings.hnsw_ef_construction,\n    m=settings.hnsw_m\n)\n\ndoc_ids, X = vector_store.get_all_doc_embeddings()\nprint(f"Loaded {len(X)} document embeddings.")\n\n# 2. Load Models\nwith open(settings.artifacts_dir / "fcm_model.pkl", 'rb') as f:\n    fcm_model = pickle.load(f)\n    \nwith open(settings.artifacts_dir / "cluster_thresholds.pkl", 'rb') as f:\n    thresholds = pickle.load(f)\n    \nprint(f"Loaded FCM model with K={fcm_model.K}")'''
    cells.append(create_cell("code", load_code))

    cells.append(create_cell(
        "markdown", "## 1. UMAP Projection of Fuzzy Clusters"))

    umap_code = '''# Compute UMAP reduction\nreducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)\nembedding_2d = reducer.fit_transform(X)\n\n# Extrat crisp labels for coloring\nlabels = np.argmax(fcm_model.U, axis=1)\n\n# Plot\nplt.figure(figsize=(12, 8))\nscatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab20', s=5, alpha=0.6)\nplt.title(f"UMAP Projection of 20 Newsgroups (K={fcm_model.K})")\nplt.colorbar(scatter, label="Cluster ID")\nplt.show()'''
    cells.append(create_cell("code", umap_code))

    cells.append(create_cell("markdown", "## 2. Cross-Membership Heatmap"))

    heatmap_code = '''# Calculate cluster co-occurrence based on soft memberships\nco_membership = fcm_model.U.T @ fcm_model.U\n# Normalize row-wise\nco_membership = co_membership / np.diag(co_membership)[:, None]\n\nplt.figure(figsize=(10, 8))\nsns.heatmap(co_membership, annot=False, cmap="YlGnBu", xticklabels=True, yticklabels=True)\nplt.title("Fuzzy Cluster Cross-Membership Probability")\nplt.xlabel("Cluster ID")\nplt.ylabel("Cluster ID")\nplt.show()'''
    cells.append(create_cell("code", heatmap_code))

    cells.append(create_cell("markdown", "## 3. Theta Exploration"))
    theta_code = '''theta_values = [0.70, 0.80, 0.85, 0.92, 0.98]\nprint(f"Adaptive thresholds computed by model: {thresholds}")'''
    cells.append(create_cell("code", theta_code))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(out_dir / "analysis.ipynb", "w", encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)


if __name__ == "__main__":
    main()
