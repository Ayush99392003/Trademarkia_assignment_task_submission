"""
Ingests raw 20 Newsgroups data, cleans it, chunks it, embeds it,
and stores the results in ChromaDB collections.
"""

from src.vector_store import VectorStore
from src.embedder import Embedder
from src.preprocessor import NewsGroupPreprocessor
from src.config import settings
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
import sys
from pathlib import Path

# Add project root to sys.path to allow importing src
sys.path.append(str(Path(__file__).parent.parent))


def main():
    console = Console()
    console.print("[bold cyan]Starting Data Ingestion Pipeline[/bold cyan]")

    # 1. Initialize components
    preprocessor = NewsGroupPreprocessor(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        min_chunk_words=settings.min_chunk_words,
        min_doc_chars=settings.min_doc_chars
    )
    embedder = Embedder(model_name=settings.model_name,
                        batch_size=settings.batch_size)
    vector_store = VectorStore(
        db_dir=settings.db_dir,
        space=settings.hnsw_space,
        ef_construction=settings.hnsw_ef_construction,
        m=settings.hnsw_m
    )

    # 2. Gather files from data directory
    data_path = Path(settings.data_root)
    if not data_path.exists():
        console.print(
            f"[bold red]Error[/bold red]: Data root {data_path} not found.")
        sys.exit(1)

    all_files = [p for p in data_path.rglob('*') if p.is_file()]
    total_files = len(all_files)

    console.print(f"Found [bold]{total_files}[/bold] files in {data_path}")

    processed_count = 0
    dropped_count = 0
    total_chunks = 0

    # Using rich progress to show processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task1 = progress.add_task(
            "[cyan]Processing documents...", total=total_files)

        # Process files in batches to optimize embedding
        file_batch = []
        batch_size = 500  # Number of files to hold in memory before embedding

        for i, file_path in enumerate(all_files):
            # The category is the parent directory name
            newsgroup = file_path.parent.name
            doc_id = f"{newsgroup}_{file_path.name}"

            result = preprocessor.process_file(file_path)

            if result is None:
                dropped_count += 1
            else:
                result["doc_id"] = doc_id
                result["newsgroup"] = newsgroup
                file_batch.append(result)

            progress.advance(task1)

            # Embed and store when batch full or at end
            if len(file_batch) >= batch_size or i == total_files - 1:
                if not file_batch:
                    continue

                # Collect all chunks across all docs in the batch
                flat_chunks = []
                chunk_to_doc_idx = []

                for doc_idx, doc_data in enumerate(file_batch):
                    for chunk in doc_data["chunks"]:
                        flat_chunks.append(chunk)
                        chunk_to_doc_idx.append(doc_idx)

                if flat_chunks:
                    # Embed all chunks
                    all_chunk_embs = embedder.encode_batch(flat_chunks)

                    # Distribute back to documents
                    current_idx = 0
                    for doc_data in file_batch:
                        num_chunks = len(doc_data["chunks"])
                        if num_chunks == 0:
                            continue

                        doc_chunk_embs = all_chunk_embs[current_idx:current_idx+num_chunks]
                        current_idx += num_chunks

                        # 1. Calculate doc-level mean embedding
                        doc_emb = embedder.doc_embedding(doc_chunk_embs)

                        # 2. Add metadata
                        base_meta = {
                            "newsgroup": doc_data["newsgroup"],
                            "subject": doc_data["subject"]
                        }

                        # 3. Store doc embedding
                        vector_store.add_doc_embedding(
                            doc_id=doc_data["doc_id"],
                            embedding=doc_emb,
                            metadata=base_meta
                        )

                        # 4. Store chunk embeddings
                        vector_store.add_chunks(
                            doc_id=doc_data["doc_id"],
                            chunks=doc_data["chunks"],
                            embeddings=doc_chunk_embs,
                            base_metadata=base_meta
                        )

                        processed_count += 1
                        total_chunks += num_chunks

                file_batch = []  # clear batch

    # 3. Print Summary Table
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Files Scanned", str(total_files))
    table.add_row("Successfully Processed", str(processed_count))
    table.add_row("Dropped (Too Short/Noise)", str(dropped_count))
    table.add_row("Total Semantic Chunks", str(total_chunks))
    table.add_row("Vector Store Chunks", str(vector_store.count_chunks()))

    console.print(table)
    console.print("[bold green]Ingestion Complete![/bold green]")


if __name__ == "__main__":
    main()
