from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """Embedding model for documents and queries."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        normalize_embedding: bool = True,
        document_prefix: str = "passage: ",
        query_prefix: str = "query: ",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embedding = normalize_embedding
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Create embeddings for document chunks."""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        prepared_texts = [
            f"{self.document_prefix}{text}"
            for text in texts
        ]

        embeddings = self.model.encode(
            prepared_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embedding,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Create embeddings for one search query."""

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        prepared_query = f"{self.query_prefix}{query}"

        embedding = self.model.encode(
            prepared_query,
            normalize_embeddings=self.normalize_embedding,
            convert_to_numpy=True,
        )

        return embedding.astype(np.float32)

@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str,
    device: str,
    batch_size: str,
    normalize_embeddings: bool,
    document_prefix: str,
    query_prefix: str,
) -> EmbeddingModel:
    """Create and cache the embedding model."""

    return EmbeddingModel(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        normalize_embedding=normalize_embeddings,
        document_prefix=document_prefix,
        query_prefix=query_prefix,
    )

def create_embedding_model(
        embedding_config: dict[str, Any],
) -> EmbeddingModel:
    """Create an embedding model from project configuration."""

    return get_embedding_model(
        model_name=embedding_config["model_name"],
        device=embedding_config.get("device", "cpu"),
        batch_size=embedding_config.get("batch_size", 16),
        normalize_embeddings=embedding_config.get("normalize_embeddings", True,),
        document_prefix=embedding_config.get("document_prefix", "passage: "),
        query_prefix=embedding_config.get("query_prefix", "query: "),
    )

