from ragflow.config.config import get_config, resolve_path
from ragflow.ingestion.loader import load_pdf
from ragflow.ingestion.cleaner import clean_pages
from ragflow.ingestion.chunker import chunk_pages
from ragflow.ingestion.writer import save_chunks
from ragflow.ingestion.image_processor import create_image_chunks, extract_images

from dotenv import load_dotenv

def main() -> None:

    load_dotenv()

    config = get_config()

    raw_data_dir = resolve_path(
        config["paths"]["raw_data"]
    )

    ingestion_config = config["ingestion"]
    cleaning_config = config["cleaning"]
    chunking_config = config["chunking"]
    image_config = config["image_processing"]


    pdf_path = raw_data_dir / "sample.pdf"

    pages = load_pdf(pdf_path, supported_extensions=ingestion_config["supported_extensions"], skip_empty_pages=ingestion_config['skip_empty_pages'])

    cleaned_pages = clean_pages(pages=pages, cleaning_config=cleaning_config)

    text_chunks = chunk_pages(pages=cleaned_pages, chunking_config=chunking_config,)

    image_chunks = []

    if image_config.get("enabled", False):
        image_output_directory = resolve_path(image_config["output_directory"])

        extracted_images = extract_images(pdf_path, output_directory=image_output_directory, minimum_width=image_config["minimum_width"], minimum_height=image_config["minimum_height"])

        print(
            f"Extracted image: "
            f"{len(extracted_images)}"
        )

        if extracted_images:
            image_chunks = create_image_chunks(extracted_images, image_config=image_config)

        chunks = text_chunks + image_chunks

    



    print(f"Project: {config['project']['name']}")
    print(f"Loaded pages: {len(pages)}")
    print(f"Cleaned pages: {len(cleaned_pages)}")
    print(f"Text chunks: {len(text_chunks)}")
    print(f"Image chunks: {len(image_chunks)}")
    print(f"Total chunks: {len(chunks)}")

    if ingestion_config["save_processed_data"]:
        processed_data_dir = resolve_path(config["paths"]["processed_data"])

        output_path = processed_data_dir / f"{pdf_path.stem}_chunks.json"

        saved_path = save_chunks(chunks=chunks, output_path=output_path)

        print(f"Saved to: {saved_path}")

    for chunk in chunks[:20]:    
        metadata = chunk["metadata"]
        preview = chunk["text"][:400].replace("\n", " ")

        print(f"\n Chunk ID: {chunk['id']}")
        print(f"Content type: {metadata['content_type']}")
        print(f"File:{metadata['filename']}")
        print(f"Page: {metadata['page']}")
        print(f"Characters: {metadata['character_count']}")

        if metadata["content_type"] == "image":
            print(
                f"Image: "
                f"{metadata['image_path']}"
            )

        print(f"Text: {preview}")


if __name__ == "__main__":
    main()

