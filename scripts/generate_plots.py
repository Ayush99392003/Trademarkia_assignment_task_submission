from src.vector_store import VectorStore
from src.config import settings
import sys
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# Add project root to sys.path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))


# Plotting config
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("1. Loading Data...")
vector_store = VectorStore(
    db_dir=settings.db_dir,
    space=settings.hnsw_space,
    ef_construction=settings.hnsw_ef_construction,
    m=settings.hnsw_m
)

doc_ids, X = vector_store.get_all_doc_embeddings()
print(f"Loaded {len(X)} embeddings.")

print("2. Loading FCM Model...")
with open(settings.artifacts_dir / "fcm_model.pkl", 'rb') as f:
    fcm_model = pickle.load(f)

print("3. Computing UMAP (this will take ~10-15 seconds)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                    metric='cosine', random_state=42)
embedding_2d = reducer.fit_transform(X)
labels = np.argmax(fcm_model.U, axis=1)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab20', s=5, alpha=0.6)
plt.title(f"UMAP Projection of 20 Newsgroups (K={fcm_model.K})")
plt.colorbar(scatter, label="Cluster ID")
plt.savefig(settings.artifacts_dir / "umap_clusters.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("Saved umap_clusters.png!")

print("4. Computing Cross-Membership Heatmap...")
co_membership = fcm_model.U.T @ fcm_model.U
co_membership = co_membership / np.diag(co_membership)[:, None]

plt.figure(figsize=(10, 8))
sns.heatmap(co_membership, annot=False, cmap="YlGnBu",
            xticklabels=True, yticklabels=True)
plt.title("Fuzzy Cluster Cross-Membership Probability")
plt.xlabel("Cluster ID")
plt.ylabel("Cluster ID")
plt.savefig(settings.artifacts_dir / "heatmap.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("Saved heatmap.png!")
print("All visuals generated successfully in the artifacts folder!")
