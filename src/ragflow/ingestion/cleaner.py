import re
import unicodedata
from typing import Any

def clean_text(
        text: str,
        *,
        unicode_form: str = "NFC",
        remove_null_characters: bool = True,
        normalize_spaces: bool = True,
        preserve_paragraphs: bool = True,
) -> str:
    """Clean extracted text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if remove_null_characters:
        text = text.replace("\x00", "")

    text = unicodedata.normalize(unicode_form, text)

    #Remove white space at beginning and end of each line
    text = "\n".join(
        line.strip()
        for line in text.split("\n")
    )

    if normalize_spaces:
        #Replace repeated spaces without removing new lines
        text = re.sub(r"[^\S\n]+", " ", text)

    if preserve_paragraphs:
        #Allow no more than one blank line between paragraphs
        text = re.sub(r"\n{3,}", "\n\n", text)

    else:
        #Combine all lines in one continuous text
        text = re.sub(r"\s+", " ", text)

    return text.strip()

def clean_pages(
    pages: list[dict[str, Any]],
    cleaning_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Clean extracted pages."""

    cleaned_pages = []

    for page in pages:
        cleaned_text = clean_text(
            page['text'],
            unicode_form=cleaning_config.get(
                "unicode_form",
                "NFC"
            ),
            remove_null_characters=cleaning_config.get(
                "remove_null_characters",
                True
            ),
            normalize_spaces=cleaning_config.get(
                "normalize_spaces",
                True,
            ),
            preserve_paragraphs=cleaning_config.get(
                "preserve_paragraphs",
                True,
            ),
        )

        min_text_length = cleaning_config.get(
            "min_text_length",
            20,
        )

        if len(cleaned_text) < min_text_length:
            continue

        cleaned_pages.append(
            {
                "text": cleaned_text,
                "metadata": page["metadata"].copy(),
            }
        )

    return cleaned_pages