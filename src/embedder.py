"""
SentenceTransformers wrapper using the distilled MiniLM representation.
Handles batched chunk embedding and pools doc-level vectors.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Handles batched chunk embedding and averaged doc embedding extraction.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        # We load the model, normalise internally automatically
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a large list of texts using batches.
        Always normalises so cosine similarity = doc product.
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            all_embeddings.append(embeddings)

        if not all_embeddings:
            return np.array([])
        return np.vstack(all_embeddings)

    def doc_embedding(self, chunk_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculates doc-level mean embedding directly from its chunk embeddings.
        Re-normalises the resulting vector back to unit length.
        """
        # Document level embedding is simply the mean across chunk vectors
        mean_embedding = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)

        if norm == 0:
            return mean_embedding
        return mean_embedding / norm
