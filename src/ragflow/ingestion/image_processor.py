import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

import pymupdf
import requests

def extract_images(
    pdf_path: str | Path,
    output_directory: str | Path,
    *,
    minimum_width: int = 200,
    minimum_height: int = 200,
) -> list[dict[str, Any]]:
    """Extract raster images from a PDF."""

    pdf_path = Path(pdf_path)
    output_directory = Path(output_directory)

    output_directory.mkdir(parents=True, exist_ok=True)

    extracted_images = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_images = page.get_images(full=True)

            seen_xrefs = set()

            for image_index, image_info in enumerate(page_images, start=1):
                xref = image_info[0]

                if xref in seen_xrefs:
                    continue

                seen_xrefs.add(xref)

                image_data = document.extract_image(xref)

                width = image_data["width"]
                height = image_data["height"]

                if width < minimum_width or height < minimum_height:
                    continue

                extension = image_data["ext"]

                image_name = (
                    f"{pdf_path.stem}-"
                    f"p{page_index + 1:04d}-"
                    f"img{image_index:03d}."
                    f"{extension}"
                )

                image_path = output_directory / image_name

                image_path.write_bytes(image_data["image"])

                extracted_images.append(
                    {
                        "path": image_path,
                        "metadata": {
                            "filename": pdf_path.name,
                            "source": str(pdf_path),
                            "page": page_index + 1,
                            "image_index": image_index,
                            "content_type": "image",
                            "width": width,
                            "height": height,
                        },
                    }
                )

    return extracted_images

def encode_image(
    image_path: str | Path,
) -> str:
    """Encode a local image as a Base64 data URL."""

    image_path = Path(image_path)

    mime_type, _ = mimetypes.guess_type(image_path.name)

    if mime_type is None:
        mime_type = "image/png"

    encoded_image = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"

def describe_image(
    image_path: str | Path,
    *,
    model_name: str,
    timeout_seconds: int = 120,
    max_tokens: int = 500,
    use_cache: bool = True,
) -> str:
    """Describe a document image using OpenRouter."""

    image_path = Path(image_path)

    cache_path = image_path.with_suffix(image_path.suffix + ".txt")

    if use_cache and cache_path.exists():
        cache_description = cache_path.read_text(encoding="utf-8").strip()

        if cache_description:
            return cache_description

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY was not found.")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this document image. Return only a concise "
                            "plain-text description in one or two paragraphs, "
                            "with no Markdown headings, bullet points, separators, "
                            "or introductory phrases. Start directly with the "
                            "image content. For charts, state the title, chart type, "
                            "categories, visible values, comparisons, and main trend. "
                            "For photographs, describe important people, objects, "
                            "actions, setting, and visible text. For diagrams, explain "
                            "the components and their relationships. Do not invent "
                            "information. Limit the response to 500 words."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image(image_path),
                        },
                    },
                ],
            }
        ],
        "reasoning": {
            "effort": "none",
        },
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenRouter image request failed: "
            f"{response.status_code}\n"
            f"{response.text}"
        )

    response_data = response.json()

    choices = response_data.get("choices", [])

    if not choices:
        raise RuntimeError(
            "OpenRouter returned no image description."
        )

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        description = content.strip()

    elif isinstance(content, list):
        description = " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        ).strip()

    else:
        description = ""

    if not description:
        finish_reason = choice.get(
            "finish_reason",
            "unknown",
        )

        reasoning = (
            message.get("reasoning")
            or message.get("reasoning_content")
        )

        reasoning_length = (
            len(reasoning)
            if isinstance(reasoning, str)
            else 0
        )

        raise RuntimeError(
            "OpenRouter returned no visible description. "
            f"finish_reason={finish_reason}, "
            f"reasoning_length={reasoning_length}. "
            "Try disabling reasoning or increasing max_tokens."
        )

    if use_cache:
        cache_path.write_text(
            description,
            encoding="utf-8",
        )

    return description

def create_image_chunks(
    extracted_images: list[dict[str, Any]],
    image_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert extracted images into searchable text chunks."""

    image_chunks = []

    for image in extracted_images:
        image_path = image["path"]
        metadata = image["metadata"]

        try:
            description = describe_image(
                image_path=image_path,
                model_name=image_config["vision_model"],
                timeout_seconds=image_config.get(
                    "timeout_seconds",
                    120
                ),
                max_tokens=image_config.get(
                    "max_tokens",
                    1000,
                ),
                use_cache=image_config.get(
                    "use_cache",
                    True,
                ),
            )

        except RuntimeError as error:
            print(
                f"Skipped image {image_path.name}: "
                f"{error}"
            )

            continue

        chunk_id = (
            f"{metadata['filename']}-"
            f"p{metadata['page']:04d}-"
            f"img{metadata['image_index']:03d}"
        )

        chunk_text = (
            f"Image description from "
            f"{metadata['filename']}, "
            f"page {metadata['page']}:\n"
            f"{description}"
        )

        image_metadata = {
            **metadata,
            "chunk_id": chunk_id,
            "image_path": str(image_path),
            "character_count": len(chunk_text),
        }

        image_chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "metadata": image_metadata,
            }
        )

    return image_chunks

