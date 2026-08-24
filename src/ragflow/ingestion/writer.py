import json
from pathlib import Path
from typing import Any

def save_chunks(
    chunks: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save processed chunks to a JSON file."""

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".json":
        raise ValueError(
            "Output file must have a .json extension."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_chunks": len(chunks),
        "chunks": chunks
    }

    with output_path.open("w", encoding="utf-8",) as file:
        json.dump(data, file, indent=2)

    return output_path