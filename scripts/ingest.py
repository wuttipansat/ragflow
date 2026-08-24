from ragflow.config.config import get_config, resolve_path
from ragflow.ingestion.loader import load_pdf
from ragflow.ingestion.cleaner import clean_pages

def main() -> None:

    config = get_config()

    raw_data_dir = resolve_path(
        config["paths"]["raw_data"]
    )

    ingestion_config = config["ingestion"]
    cleaning_config = config["cleaning"]


    pdf_path = raw_data_dir / "sample.pdf"
    pages = load_pdf(pdf_path, supported_extensions=ingestion_config["supported_extensions"], skip_empty=ingestion_config['skip_empty_pages'])
    cleaned_pages = clean_pages(pages=pages, cleaning_config=cleaning_config)

    print(f"Project: {config['project']['name']}")
    print(f"Loaded pages: {len(pages)}")
    print(f"Cleaned pages: {len(cleaned_pages)}")

    for page in cleaned_pages[:3]:
        metadata = page["metadata"]
        text_preview = page["text"][:300].replace("\n", " ")

        print(f"\n File:{metadata['filename']}")
        print(f"Pages: {metadata['page']}")
        print(f"Text: {text_preview}")


if __name__ == "__main__":
    main()

