from ragflow.config.config import get_config, resolve_path
from ragflow.embeddings.embedding import create_embedding_model
from ragflow.retrieval.vector_store import ChromaVectorStore

def main() -> None:
    config = get_config()

    query = "What are his computer vision skills?"

    embedding_model = create_embedding_model(config["embedding"])

    query_embedding = embedding_model.embed_query(query)

    vector_store = ChromaVectorStore(
        presist_directory=resolve_path(config["paths"]["vector_store"]),
        collection_name=config["vector_store"]["collection_name"],
        distance_metric=config["vector_store"]["distance_metric"],
    )

    results = vector_store.search(
        query_embedding=query_embedding,
    )

    for index, document in enumerate(
        results["documents"][0],
        start=1,
    ):
        distance = results["distances"][0][index - 1]
        metadata = results["metadatas"][0][index - 1]

        similarity = 1 - distance

        print()
        print(f"Rank: {index}")
        print(f"Similarity: {similarity:.4f}")
        print(f"File: {metadata['filename']}")
        print(f"Page: {metadata['page']}")
        print(f"Text: {document[:300]}")

if __name__ == "__main__":
    main()