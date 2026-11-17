from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from unstructured.partition.pdf import partition_pdf


@dataclass
class DocElement:
    element_type: Literal["text", "image"]
    content: str
    source_pdf: str
    page_number: int


def parse_pdf(pdf_path: Path, image_output_dir: Path) -> list[DocElement]:
    image_output_dir.mkdir(parents=True, exist_ok=True)

    raw_elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_output_dir=str(image_output_dir),
    )

    elements: list[DocElement] = []
    for el in raw_elements:
        page_number = el.metadata.page_number or 1
        image_path = getattr(el.metadata, "image_path", None)
        if image_path:
            elements.append(
                DocElement(
                    element_type="image",
                    content=str(Path(image_path).resolve()),
                    source_pdf=pdf_path.name,
                    page_number=page_number,
                )
            )
        elif str(el).strip():
            elements.append(
                DocElement(
                    element_type="text",
                    content=str(el),
                    source_pdf=pdf_path.name,
                    page_number=page_number,
                )
            )

    return elements
