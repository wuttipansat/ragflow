from ragflow.config.config import get_config, resolve_path
from ragflow.ingestion.loader import load_pdf
from ragflow.ingestion.cleaner import clean_pages
from ragflow.ingestion.chunker import chunk_pages

def main() -> None:

    config = get_config()

    raw_data_dir = resolve_path(
        config["paths"]["raw_data"]
    )

    ingestion_config = config["ingestion"]
    cleaning_config = config["cleaning"]
    chunking_config = config["chunking"]


    pdf_path = raw_data_dir / "sample.pdf"
    pages = load_pdf(pdf_path, supported_extensions=ingestion_config["supported_extensions"], skip_empty=ingestion_config['skip_empty_pages'])
    cleaned_pages = clean_pages(pages=pages, cleaning_config=cleaning_config)
    chunks = chunk_pages(pages=cleaned_pages, chunking_config=chunking_config,)

    print(f"Project: {config['project']['name']}")
    print(f"Loaded pages: {len(pages)}")
    print(f"Cleaned pages: {len(cleaned_pages)}")
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks[:10]:    
        metadata = chunk["metadata"]
        preview = chunk["text"][:400].replace("\n", " ")

        print(f"\n Chunk ID: {chunk['id']}")
        print(f"File:{metadata['filename']}")
        print(f"Page: {metadata['page']}")
        print(f"Characters: {metadata['character_count']}")
        print(f"Text: {preview}")


if __name__ == "__main__":
    main()

