from typing import Any

def split_text(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[str]:
    """Split text into overlapping chunks without cutting words."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be between 0 and chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(
            start + chunk_size,
            len(text),
        )

        if end < len(text):
            while (
                end > start
                and not text[end - 1].isspace()
            ):
                end -= 1

        if end == start:
            end = min(
                start + chunk_size,
                len(text),
            )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(
            end - chunk_overlap,
            0,
        )

        while (
            next_start > start
            and not text[next_start - 1].isspace()
        ):
            next_start -= 1


        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

def chunk_pages(
        pages: list[dict[str, Any]],
        chunking_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split cleaned pages into chunks."""

    chunk_size = chunking_config.get(
        "chunk_size",
        800,
    )

    chunk_overlap = chunking_config.get(
        "chunk_overlap",
        150,
    )

    min_chunk_size = chunking_config.get(
        "min_chunk_size",
        50,
    )

    chunks = []

    for page in pages:
        page_chunks = split_text(
            page['text'],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        metadata = page["metadata"]
        filename = metadata['filename']
        page_number = metadata['page']

        for chunk_index, chunk_text in enumerate(page_chunks):
            if len(chunk_text) < min_chunk_size:
                continue

            chunk_id = (
                f"{filename}"
                f"p{page_number:04d}-"
                f"c{chunk_index:04d}"
            )

            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "character_count": len(chunk_text),
                }
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )


    return chunks