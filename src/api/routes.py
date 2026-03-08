"""
Core API routing module exposing the search interface and cached telemetry.
"""

import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import numpy as np
import sklearn.metrics.pairwise
from scipy.stats import entropy

from src.cache import CacheEntry

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    cluster_theta: Optional[float] = None
    result: Dict[str, Any]
    dominant_cluster: int
    membership_entropy: float
    search_clusters_used: List[int]
    latency_ms: float


@router.post("/query", response_model=SearchResult)
async def semantic_search(request: Request, payload: QueryRequest):
    start_time = time.time()

    # Access state singletons
    embedder = request.app.state.embedder
    fcm_model = request.app.state.fcm_model
    cache = request.app.state.cache
    graph = request.app.state.cluster_graph
    vector_store = request.app.state.vector_store

    if not fcm_model:
        raise HTTPException(
            status_code=503, detail="Clustering artifacts missing. Run pipeline first.")

    # 1. Embed user query text
    query_text = payload.query
    query_emb = embedder.encode_batch([query_text])[0]

    # 2. Extract mathematically derived soft-memberships
    membership_vec = fcm_model.predict(query_emb.reshape(1, -1))[0]
    dominant_cluster = int(np.argmax(membership_vec))

    # Evaluate explicit neighbors directly
    graph_neighbours = graph.get_search_clusters(
        query_emb, fcm_model.centroids, top_k=3)
    graph_neighbours = [n for n in graph_neighbours if n != dominant_cluster]

    # 3. Cache lookup
    hit_entry, clusters_checked = cache.lookup(
        query_emb, membership_vec, graph_neighbours)

    if hit_entry is not None:
        # Cache Hit!
        latency = (time.time() - start_time) * 1000
        sim = float(sklearn.metrics.pairwise.cosine_similarity(
            query_emb.reshape(1, -1),
            hit_entry.query_embedding.reshape(1, -1)
        )[0][0])

        return SearchResult(
            query=query_text,
            cache_hit=True,
            matched_query=hit_entry.query_text,
            similarity_score=sim,
            cluster_theta=cache._get_theta(hit_entry.dominant_cluster),
            result=hit_entry.result,
            dominant_cluster=hit_entry.dominant_cluster,
            membership_entropy=hit_entry.entropy,
            search_clusters_used=clusters_checked,
            latency_ms=latency
        )

    # 4. Cache Miss - Execute ChromaDB query
    results = vector_store.query_chunks(
        query_embedding=query_emb,
        cluster_ids=clusters_checked,
        n_results=payload.top_k * 2  # Oversample for re-ranking
    )

    # 5. Format and custom rerank logic
    formatted_results = {"top_chunks": []}

    if results["ids"] and len(results["ids"]) > 0:
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        chunk_list = []
        for i in range(len(ids)):
            sim = 1.0 - distances[i]
            meta = metadatas[i]

            # Simple weighting bounded by native membership alignment
            chunk_membership = meta.get("dominant_membership", 1.0)
            adjusted_score = sim * (0.8 + 0.2 * chunk_membership)

            chunk_list.append({
                "text": documents[i],
                "newsgroup": meta.get("newsgroup"),
                "subject": meta.get("subject"),
                "score": float(adjusted_score),
                "document_id": meta.get("doc_id"),
                "chunk_index": meta.get("chunk_idx")
            })

        chunk_list = sorted(chunk_list, key=lambda x: -
                            x["score"])[:payload.top_k]
        formatted_results["top_chunks"] = chunk_list

    # 6. Store cache miss
    current_entropy = float(entropy(np.clip(membership_vec, 1e-10, 1.0)))

    new_entry = CacheEntry(
        query_text=query_text,
        query_embedding=query_emb,
        result=formatted_results,
        dominant_cluster=dominant_cluster,
        membership_vec=membership_vec,
        entropy=current_entropy
    )
    cache.store(new_entry)

    latency = (time.time() - start_time) * 1000

    return SearchResult(
        query=query_text,
        cache_hit=False,
        result=formatted_results,
        dominant_cluster=dominant_cluster,
        membership_entropy=current_entropy,
        search_clusters_used=clusters_checked,
        latency_ms=latency
    )


@router.get("/cache/stats")
async def get_cache_stats(request: Request):
    return request.app.state.cache.stats()


@router.delete("/cache")
async def flush_cache(request: Request):
    request.app.state.cache.flush()
    return {"status": "success", "message": "Semantic cache flushed."}
