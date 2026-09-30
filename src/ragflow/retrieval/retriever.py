from typing import Any

from ragflow.embeddings.embedding import EmbeddingModel
from ragflow.retrieval.vector_store import ChromaVectorStore

class Retriever:
    """Retrieve relevant document chunks for a query."""

    def __init__(
            self,
            embedding_model: EmbeddingModel,
            vector_store: ChromaVectorStore,
            *,
            top_k: int = 3,
            minimum_similarity: float = 0.30,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k
        self.minimum_similarity = minimum_similarity

    def retrieve(
        self,
        query: str,
    ) -> list[dict]:
        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )