import json

import numpy as np

from ragflow.config.config import get_config, resolve_path
from ragflow.embeddings.embedding import create_embedding_model

def main() -> None:
    config = get_config()

    processed_dir = resolve_path(config["paths"]["processed_data"])

    input_path = processed_dir / "sample_chunks.json"

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = data["chunks"]

    if not chunks:
        raise ValueError("No chunks were found.")

    texts = [chunk["text"] for chunk in chunks]

    embedding_model = create_embedding_model(config["embedding"])

    document_embeddings = embedding_model.embed_documents(texts)

    query = "What computer vision experience does he have?"

    query_embedding = embedding_model.embed_query(query)

    scores = document_embeddings @ query_embedding

    best_indices = np.argsort(scores)[::-1][:3]

    print(f"Document embeddings: {document_embeddings.shape}")
    print(f"Query embedding: {query_embedding.shape}")
    print(f"\nQuestion: {query}")

    for rank, index in enumerate(best_indices, start=1):
        chunk = chunks[index]
        score = float(scores[index])

        preview = chunk["text"][:300]

        if len(chunk["text"]) > 300:
            preview = preview.rsplit(" ", maxsplit=1)[0] + "..."

        print()
        print(f"Rank: {rank}")
        print(f"Score: {score:.4f}")
        print(f"Chunk ID: {chunk['id']}")
        print(f"Page: {chunk['metadata']['page']}")
        print(f"Text: {preview}")

if __name__ == "__main__":
    main()