"""
ChromaDB vector store implementation configured with local HNSW filters.
Enables segregation of queries to bounded fuzzy cluster subsets.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np


class VectorStore:
    """
    Interface connecting to ChromaDB for efficient retrieval.
    Holds chunks collection for search and doc collection for FCM clustering.
    """

    def __init__(self, db_dir: Path, space: str = "cosine",
                 ef_construction: int = 200, m: int = 16):

        self.client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # HNSW tuning overrides
        hnsw_metadata = {
            "hnsw:space": space,
            "hnsw:construction_ef": ef_construction,
            "hnsw:M": m
        }

        # The primary search collection (granularity: chunk)
        self.chunk_coll = self.client.get_or_create_collection(
            name="chunks",
            metadata=hnsw_metadata
        )

        # The clustering foundation collection (granularity: doc)
        self.doc_coll = self.client.get_or_create_collection(
            name="doc_embeddings",
            metadata=hnsw_metadata
        )

    def add_chunks(self, doc_id: str, chunks: List[str],
                   embeddings: np.ndarray, base_metadata: Dict[str, Any]):
        """
        Inserts all semantic chunks associated with a document.
        """
        n_chunks = len(chunks)
        ids = [f"{doc_id}_chunk_{i}" for i in range(n_chunks)]

        metadatas = []
        for i in range(n_chunks):
            meta = base_metadata.copy()
            meta.update({
                "doc_id": doc_id,
                "chunk_idx": i,
                "total_chunks": n_chunks,
                "cluster_id": -1,  # -1 represents an uninitialised state
                "dominant_membership": 0.0
            })
            metadatas.append(meta)

        self.chunk_coll.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def add_doc_embedding(self, doc_id: str, embedding: np.ndarray,
                          metadata: Dict[str, Any]):
        """
        Records the overall unit-normalised document embedding.
        """
        self.doc_coll.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

    def get_all_doc_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Outputs the bulk array (N x 384) to be consumed by mathematical FCM.
        """
        result = self.doc_coll.get(include=["embeddings"])
        if not result["ids"]:
            return [], np.array([])
        return result["ids"], np.array(result["embeddings"])

    def query_chunks(self, query_embedding: np.ndarray,
                     cluster_ids: List[int], n_results: int = 10) -> dict:
        """
        Targets a specifically bounded local HNSW sub-graph within designated clusters.
        """
        where_filter = None
        if cluster_ids:
            if len(cluster_ids) == 1:
                where_filter = {"cluster_id": cluster_ids[0]}
            else:
                where_filter = {"cluster_id": {"$in": cluster_ids}}

        results = self.chunk_coll.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter
        )

        return results

    def update_cluster_metadata(self, doc_id: str, cluster_id: int,
                                dominant_membership: float):
        """
        Overwrites chunk metadata locally after successful convergence of FCM.
        """
        results = self.chunk_coll.get(
            where={"doc_id": doc_id},
            include=["metadatas"]
        )

        if not results["ids"]:
            return

        updated_metadatas = []
        for meta in results["metadatas"]:
            new_meta = meta.copy()
            new_meta["cluster_id"] = int(cluster_id)
            new_meta["dominant_membership"] = float(dominant_membership)
            updated_metadatas.append(new_meta)

        self.chunk_coll.update(
            ids=results["ids"],
            metadatas=updated_metadatas
        )

    def count_chunks(self) -> int:
        return self.chunk_coll.count()

    def count_docs(self) -> int:
        return self.doc_coll.count()
