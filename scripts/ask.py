from dotenv import load_dotenv

from ragflow.config.config import get_config, resolve_path
from ragflow.embeddings.embedding import (
    create_embedding_model,
)
from ragflow.generation.generator import OpenRouterGenerator
from ragflow.generation.prompt import build_rag_prompt
from ragflow.retrieval.retriever import Retriever
from ragflow.retrieval.vector_store import ChromaVectorStore


def main() -> None:
    load_dotenv()

    config = get_config()


    embedding_model = create_embedding_model(
        config["embedding"]
    )


    vector_config = config["vector_store"]

    vector_store = ChromaVectorStore(
        persist_directory=resolve_path(
            config["paths"]["vector_store"]
        ),
        collection_name=vector_config["collection_name"],
        distance_metric=vector_config["distance_metric"],
    )


    retrieval_config = config["retrieval"]

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        top_k=retrieval_config["top_k"],
        minimum_similarity=retrieval_config[
            "minimum_similarity"
        ],
    )


    generation_config = config["generation"]

    generator = OpenRouterGenerator(
        model_name=generation_config["model_name"],
        base_url=generation_config["base_url"],
        temperature=generation_config["temperature"],
        max_tokens=generation_config["max_tokens"],
        timeout_seconds=generation_config[
            "timeout_seconds"
        ],
    )


    query = input("Question: ").strip()

    if not query:
        raise ValueError("Question cannot be empty.")

    retrieved_chunks = retriever.retrieve(query)

    if not retrieved_chunks:
        print(
            "No sufficiently relevant information "
            "was found in the document."
        )
        return

    prompt = build_rag_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    print("\nGenerating answer...")

    answer = generator.generate(prompt)

    print()
    print("Answer")
    print("------")
    print(answer)

    print()
    print(f"Model used: {generator.last_model_used}")

    print()
    print("Retrieved sources")
    print("-----------------")

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk["metadata"]

        print(
            f"[Source {index}] "
            f"{metadata.get('filename', 'unknown')}, "
            f"page {metadata.get('page', 'unknown')}, "
            f"similarity={chunk['similarity']:.4f}"
        )


if __name__ == "__main__":
    main()