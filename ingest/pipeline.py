import logging
import sys
from pathlib import Path

from ingest.pdf_parser import parse_pdf
from ingest.store import add_elements

logger = logging.getLogger(__name__)


def ingest_folder(pdf_folder: Path, vectorstore, embedder, image_output_dir: Path) -> int:
    ingested_count = 0
    for pdf_path in sorted(Path(pdf_folder).glob("*.pdf")):
        try:
            elements = parse_pdf(pdf_path, image_output_dir=image_output_dir)
            add_elements(vectorstore, embedder, elements)
        except Exception:
            logger.warning("Failed to ingest %s, skipping", pdf_path, exc_info=True)
            continue

        ingested_count += 1
        logger.info("Ingested %s (%d elements)", pdf_path.name, len(elements))

    return ingested_count


def main() -> None:
    import os

    from dotenv import load_dotenv

    from ingest.embedder import get_embedder
    from ingest.store import get_vectorstore

    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 2:
        print("Usage: python -m ingest.pipeline <pdf_folder>")
        sys.exit(1)

    pdf_folder = Path(sys.argv[1])
    database_url = os.environ["DATABASE_URL"]
    image_output_dir = Path(os.environ.get("IMAGE_OUTPUT_DIR", "data/images"))

    embedder = get_embedder()
    vectorstore = get_vectorstore(embedder, database_url)

    count = ingest_folder(pdf_folder, vectorstore, embedder, image_output_dir)
    print(f"Ingested {count} PDF(s) from {pdf_folder}")


if __name__ == "__main__":
    main()
