import json

from ragflow.config.config import get_config, resolve_path
from ragflow.embeddings.embedding import create_embedding_model
from ragflow.retrieval.vector_store import ChromaVectorStore

def main() -> None:
    config = get_config()

    processed_dir = resolve_path(config["paths"]["processed_data"])

    vector_store_dir = resolve_path(config["paths"]["vector_store"])

    input_path = processed_dir / "sample_chunks.json"

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = data["chunks"]

    if not chunks:
        raise ValueError("No chunks were found.")

    embedding_model = create_embedding_model(config["embedding"])

    texts = [chunk['text'] for chunk in chunks]

    embeddings = embedding_model.embed_documents(texts)

    vector_store = ChromaVectorStore(
        presist_directory=vector_store_dir,
        collection_name=config["vector_store"]["collection_name"],
        distance_metric=config["vector_store"]["distance_metric"],
    )

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    print(f"Indexed chunks: {len(chunks)}")
    print(
        f"Collection count: "
        f"{vector_store.collection.count()}"
    )


if __name__ == "__main__":
    main()
