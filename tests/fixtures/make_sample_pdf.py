from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

FIXTURES_DIR = Path(__file__).parent


def make_sample_image(path: Path) -> None:
    from PIL import Image

    image = Image.new("RGB", (100, 100), color=(255, 0, 0))
    image.save(str(path), format="PNG")


def make_sample_pdf() -> None:
    image_path = FIXTURES_DIR / "sample_image.png"
    make_sample_image(image_path)

    pdf_path = FIXTURES_DIR / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(72, 700, "This is a test document about multimodal RAG.")
    c.drawImage(str(image_path), 72, 500, width=100, height=100)
    c.showPage()
    c.save()


if __name__ == "__main__":
    make_sample_pdf()
