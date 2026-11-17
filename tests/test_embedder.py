from pathlib import Path

from ingest.embedder import embed_image_path, embed_text_query, get_embedder

FIXTURE_IMAGE = Path(__file__).parent / "fixtures" / "sample_image.png"


def test_embed_text_and_image_share_dimension():
    embedder = get_embedder()

    text_vector = embed_text_query(embedder, "a red square")
    image_vector = embed_image_path(embedder, str(FIXTURE_IMAGE))

    assert len(text_vector) == len(image_vector)
    assert len(text_vector) > 0
    assert isinstance(text_vector[0], float)
