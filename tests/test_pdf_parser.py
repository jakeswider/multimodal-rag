from pathlib import Path

from ingest.pdf_parser import parse_pdf

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_parse_pdf_extracts_text_and_image(tmp_path):
    elements = parse_pdf(FIXTURE_PDF, image_output_dir=tmp_path)

    text_elements = [e for e in elements if e.element_type == "text"]
    image_elements = [e for e in elements if e.element_type == "image"]

    assert any("multimodal RAG" in e.content for e in text_elements)
    assert len(image_elements) >= 1
    assert Path(image_elements[0].content).exists()
    assert all(e.source_pdf == "sample.pdf" for e in elements)
    assert all(e.page_number == 1 for e in elements)
