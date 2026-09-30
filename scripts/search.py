from ragflow.config.config import get_config, resolve_path
from ragflow.embeddings.embedding import create_embedding_model
from ragflow.retrieval.vector_store import ChromaVectorStore

def main() -> None:
    config = get_config()

    query = "What is the amount allocated to housing?"

    embedding_model = create_embedding_model(config["embedding"])

    query_embedding = embedding_model.embed_query(query)

    vector_store = ChromaVectorStore(
        persist_directory=resolve_path(config["paths"]["vector_store"]),
        collection_name=config["vector_store"]["collection_name"],
        distance_metric=config["vector_store"]["distance_metric"],
    )

    results = vector_store.search(
        query_embedding=query_embedding,
    )

    for index, result in enumerate(results, start=1):

        metadata = result.get("metadata") or {}

        print()
        print(f"Rank: {index}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"File: {metadata['filename']}")
        print(f"Page: {metadata['page']}")
        print(f"Content type: {metadata['content_type']}")
        print(f"Text: {result['text'][:300]}")

if __name__ == "__main__":
    main()