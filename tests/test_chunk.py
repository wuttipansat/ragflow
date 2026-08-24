import pytest

from ragflow.ingestion.chunker import chunk_pages, split_text

def test_one_chunk() -> None:
    text = "This is a short document."

    chunks = split_text(text, chunk_size=100, chunk_overlap=20,)

    assert chunks == [text]

def test_multiple_chunk() -> None:

    text = "This is a document." * 100

    chunks = split_text(text, chunk_size=200, chunk_overlap=50,)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)

def test_empty() -> None:

    chunks = split_text("", chunk_size=100, chunk_overlap=20,)

    assert chunks == []

def test_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text(
            "Test",
            chunk_size=100,
            chunk_overlap=100,
        )


