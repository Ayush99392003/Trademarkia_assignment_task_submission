"""
Configuration module for Trademarkia Semantic Search.
Follows PEP 8 styling and groups all hyperparameters in a single dataclass.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # Directory paths
    project_root: Path = Path(__file__).parent.parent

    # Check if running in Docker by looking for our mapped volume
    _in_docker = Path("/app/data").exists()

    data_root: Path = Path("/app/data") if _in_docker else Path(
        r"c:\Users\ayush\Videos\Trademarkia\twenty+newsgroups\20_newsgroups")
    db_dir: Path = project_root / "artifacts" / "chroma_db"
    artifacts_dir: Path = project_root / "artifacts"

    # Preprocessing & Chunking tunables
    chunk_size: int = 100
    chunk_overlap: int = 25
    min_chunk_words: int = 20
    min_doc_chars: int = 20

    # Embedding Model tunables
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    embedding_dim: int = 384

    # Vector Store (ChromaDB HNSW)
    hnsw_space: str = "cosine"
    hnsw_ef_construction: int = 200
    hnsw_m: int = 16

    # Clustering (FCM) tunables
    fcm_m_fuzziness: float = 2.0
    fcm_max_iter: int = 150
    fcm_tol: float = 1e-4
    k_range: list[int] = field(default_factory=lambda: [5, 7, 10, 12, 15, 18])

    # Semantic Cache tunables
    cache_max_size: int = 1000
    cache_base_theta: float = 0.82


settings = Settings()

# Ensure critical artifact directories exist
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
settings.db_dir.mkdir(parents=True, exist_ok=True)
