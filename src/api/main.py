"""
FastAPI application entrypoint.
Manages singleton lifespans to prevent reloading artifacts per request.
"""

from contextlib import asynccontextmanager
import pickle

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.cluster_graph import ClusterGraph
from src.cache import SemanticCache

# Import routing modules below to avoid circular dependency
from src.api.routes import router


def load_artifact(filename: str):
    path = settings.artifacts_dir / filename
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup computations
    app.state.embedder = Embedder(model_name=settings.model_name)
    app.state.vector_store = VectorStore(
        db_dir=settings.db_dir,
        space=settings.hnsw_space,
        ef_construction=settings.hnsw_ef_construction,
        m=settings.hnsw_m
    )

    app.state.fcm_model = load_artifact("fcm_model.pkl")

    # Load cluster graph
    graph = ClusterGraph()
    graph_path = settings.artifacts_dir / "cluster_graph.pkl"
    if graph_path.exists():
        graph.load(graph_path)
    app.state.cluster_graph = graph

    # Setup cache
    cache = SemanticCache(max_size=settings.cache_max_size)
    thresholds = load_artifact("cluster_thresholds.pkl")
    if thresholds:
        cache.load_thresholds(thresholds)
    app.state.cache = cache

    yield
    # Shutdown (ChromaDB handles file persistence automatically)

app = FastAPI(
    title="Trademarkia Semantic Search",
    description="Fuzzy clustered cache-backed semantic vector search.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# --- Frontend Mount ---
static_dir = settings.project_root / "src" / "api" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_frontend():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend UI file not found."}
