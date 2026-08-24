from pathlib import Path
from ragflow.config.config import get_config
from collections.abc import Sequence
import fitz

def load_pdf(file_path: str | Path, supported_extensions: Sequence[str], skip_empty: bool = True) -> list[dict]:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extensions = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in supported_extensions
    }

    if file_path.suffix.lower() not in extensions:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported types: {sorted(extensions)}"
            )

    pages = []

    with fitz.open(file_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text")

            if skip_empty and not text.strip():
                continue

            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "filename": file_path.name,
                        "source": str(file_path),
                        "page": page_index + 1
                    }
                }
            )

    return pages

config = get_config()
config["ingestion"]["supported_extensions"]

