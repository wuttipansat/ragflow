from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pymupdf


def table_to_text(
    rows: list[list[Any]],
) -> str:
    """Convert table rows into embedding-friendly text."""

    if not rows or len(rows) < 2:
        return ""

    raw_headers = rows[0]

    headers = [
        str(value).strip()
        if value is not None and str(value).strip()
        else f"Column {index + 1}"
        for index, value in enumerate(raw_headers)
    ]

    text_rows = []

    for row in rows[1:]:
        fields = []

        for header, value in zip(headers, row):
            if value is None:
                continue

            value = str(value).strip()

            if value:
                fields.append(
                    f"{header}: {value}"
                )

        if fields:
            text_rows.append("; ".join(fields))

    return "\n".join(text_rows)


def extract_tables(
    page: pymupdf.Page,
) -> list[str]:
    """Extract tables from one PDF page."""

    extracted_tables = []

    table_finder = page.find_tables()

    for table_index, table in enumerate(
        table_finder.tables,
        start=1,
    ):
        rows = table.extract()
        table_text = table_to_text(rows)

        if table_text:
            extracted_tables.append(
                f"Table {table_index}\n{table_text}"
            )

    return extracted_tables


def load_pdf(
    file_path: str | Path,
    supported_extensions: Sequence[str],
    skip_empty_pages: bool = True,
) -> list[dict[str, Any]]:
    """Load PDF text and table data by page."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    normalized_extensions = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in supported_extensions
    }

    if file_path.suffix.lower() not in normalized_extensions:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported types: {sorted(normalized_extensions)}"
        )

    pages = []

    with pymupdf.open(file_path) as document:
        for page_index, page in enumerate(document):
            page_text = page.get_text("text").strip()

            table_texts = extract_tables(page)

            combined_parts = []

            if page_text:
                combined_parts.append(page_text)

            if table_texts:
                combined_parts.append(
                    "STRUCTURED TABLE DATA\n\n"
                    + "\n\n".join(table_texts)
                )

            combined_text = "\n\n".join(
                combined_parts
            ).strip()

            if skip_empty_pages and not combined_text:
                continue

            pages.append(
                {
                    "text": combined_text,
                    "metadata": {
                        "filename": file_path.name,
                        "source": str(file_path),
                        "file_type": file_path.suffix.lower(),
                        "page": page_index + 1,
                        "table_count": len(table_texts),
                    },
                }
            )

    return pages