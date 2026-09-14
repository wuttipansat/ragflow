from ragflow.config.config import get_config, resolve_path
from ragflow.embeddings.embedding import create_embedding_model
from ragflow.retrieval.retriever import Retriever
from ragflow.retrieval.vector_store import ChromaVectorStore

def main() -> None:
    config = get_config()

    embedding_model = create_embedding_model(config["embedding"])

    vector_store = ChromaVectorStore(
        presist_directory=resolve_path(config["paths"]["vector_store"]),
        collection_name=config["vector_store"]["collection_name"],
        distance_metric=config["vector_store"]["distance_metric"],
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=config["retrieval"]["top_k"],
        minimum_similarity=config["retrieval"]["minimum_similarity"],
    )

    query = input("Question: ").strip()

    if not query:
        raise ValueError("Question cannot be empty.")

    results = retriever.retrieve(query)

    if not results:
        print("No relevant chunks were found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print()
        print(f"Rank: {rank}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Chunk ID: {result['id']}")
        print(f"File: {metadata.get('filename')}")
        print(f"Page: {metadata.get('page')}")
        print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()