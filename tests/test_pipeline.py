import shutil
from pathlib import Path

from ingest.pipeline import ingest_folder

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


class FakeVectorstore:
    def __init__(self):
        self.calls = []

    def add_embeddings(self, texts, embeddings, metadatas):
        self.calls.append((texts, embeddings, metadatas))


class FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2]

    def embed_image(self, paths):
        return [[0.3, 0.4] for _ in paths]


def test_ingest_folder_skips_bad_pdf_and_processes_good_one(tmp_path):
    pdf_folder = tmp_path / "pdfs"
    pdf_folder.mkdir()
    shutil.copy(FIXTURE_PDF, pdf_folder / "sample.pdf")
    (pdf_folder / "broken.pdf").write_bytes(b"not a real pdf")

    vectorstore = FakeVectorstore()
    embedder = FakeEmbedder()
    image_output_dir = tmp_path / "images"

    count = ingest_folder(pdf_folder, vectorstore, embedder, image_output_dir)

    assert count == 1
    assert len(vectorstore.calls) == 1
