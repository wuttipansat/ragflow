import pytest
from ragflow.ingestion.cleaner import clean_text

def test_clean_text_normalizes_spaces() -> None:
    text = "This   is   a  test."
    result = clean_text(text)

    assert result == "This is a test."

def test_clean_text_preserves_scientific_symbols() -> None:
    text = "Thickness = 100 μm and volume = 25 m³."

    result = clean_text(text, unicode_form="NFC")

    assert "100 μm" in result
    assert "25 m³" in result

def test_clean_text_preserves_paragraphs() -> None:
    text = "First paragraph.\n\n\nSecond paragraph."

    result = clean_text(text, preserve_paragraphs=True,)

    assert result == (
        "First paragraph.\n\nSecond paragraph."
    )

def test_clean_text_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        clean_text(123)