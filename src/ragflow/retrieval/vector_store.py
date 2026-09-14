from pathlib import Path
from typing import Any

import chromadb

class ChromaVectorStore:
    """Store and search document embeddings using ChromaDB."""

    def __init__(
            self,
            presist_directory: str | Path,
            collection_name: str = "documents",
            distance_metric: str = "cosine",
    ) -> None:
        self.client = chromadb.PersistentClient(
            path=str(presist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": distance_metric,
            }
        )

    def add_chunks(
            self,
            chunks: list[dict[str, Any]],
            embeddings: Any,
    ) -> None:
        """Add chunks and their embeddings to ChromaDB."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks and embeddings must match."
            )

        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                self._prepare_metadata(chunk["metadata"])
                for chunk in chunks
            ],
            embeddings=embeddings.tolist(),
        )

    def search(
            self,
            query_embedding: Any,
            top_k: int = 3,
    ) -> dict[str, Any]:
        """Search for chunks closest to the query embedding."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        return self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    @staticmethod
    def _prepare_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, str | int | float | bool]:
        """Keep only metadata types supported by ChromaDB."""

        supported_types = (str, int, float, bool)

        return {
            key: value
            for key, value in metadata.items()
            if isinstance(value, supported_types)
        }