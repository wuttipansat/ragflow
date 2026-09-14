from typing import Any

SYSTEM_PROMPT = """
You are a document question-answering assistant.

Rules:
1. Answer using only the provided document context.
2. Do not invent information.
3. If the answer is not in the context, say that the information was not found in the provided document.
4. Answer in the same language as the question.
5. Cite source using [Source 1], [Source 2], etc.
6. Keep the answer clear and concise.
""".strip()

def build_rag_prompt(
        query: str,
        retrieved_chunks: list[dict[str, Any]]
) -> str:
    """Combine the query and retrieved chunks."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not retrieved_chunks:
        raise ValueError("No retrieved chunk were provided.")

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk['metadata']

        context_parts.append(
            f"[Source {index}]\n"
            f"File: {metadata.get('filename', 'unknown')}\n"
            f"Page: {metadata.get('page', 'unknown')}\n"
            f"Context:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    return (
        f"DOCUMENT CONTEXT\n"
        "==================\n"
        f"{context}\n\n"
        f"QUESTION\n"
        "==================\n"
        f"{query}"
    )