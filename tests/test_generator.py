import base64
import io
from pathlib import Path

from langchain_core.documents import Document
from PIL import Image

from rag.generator import generate_answer, generate_answer_stream

FIXTURE_IMAGE = Path(__file__).parent / "fixtures" / "sample_image.png"


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self):
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return FakeResponse("It's a red square.")

    def stream(self, messages):
        self.last_messages = messages
        for piece in ["It's ", "a ", "red ", "square."]:
            yield FakeResponse(piece)


def test_generate_answer_builds_multimodal_prompt_and_returns_text():
    llm = FakeLLM()
    context_docs = [
        Document(page_content="A paragraph about shapes.", metadata={"element_type": "text"}),
        Document(page_content=str(FIXTURE_IMAGE), metadata={"element_type": "image"}),
    ]

    answer = generate_answer(llm, "What shape is shown?", context_docs)

    assert answer == "It's a red square."
    sent_content = llm.last_messages[0].content
    assert any(part.get("type") == "text" and "shapes" in part.get("text", "") for part in sent_content)
    image_parts = [part for part in sent_content if part.get("type") == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"].startswith("data:image/jpeg;base64,")


def test_generate_answer_downscales_oversized_images(tmp_path):
    large_image_path = tmp_path / "large.png"
    Image.new("RGB", (2000, 1000), color=(255, 0, 0)).save(large_image_path)
    original_size = large_image_path.stat().st_size

    llm = FakeLLM()
    context_docs = [
        Document(page_content=str(large_image_path), metadata={"element_type": "image"}),
    ]

    generate_answer(llm, "What shape is shown?", context_docs)

    sent_content = llm.last_messages[0].content
    image_parts = [part for part in sent_content if part.get("type") == "image_url"]
    assert len(image_parts) == 1

    data_url = image_parts[0]["image_url"]
    assert data_url.startswith("data:image/jpeg;base64,")
    encoded = data_url.removeprefix("data:image/jpeg;base64,")
    decoded_bytes = base64.b64decode(encoded)

    with Image.open(io.BytesIO(decoded_bytes)) as resized:
        assert max(resized.size) <= 512

    assert len(decoded_bytes) < original_size


def test_generate_answer_stream_yields_incremental_chunks():
    llm = FakeLLM()
    context_docs = [
        Document(page_content="A paragraph about shapes.", metadata={"element_type": "text"}),
        Document(page_content=str(FIXTURE_IMAGE), metadata={"element_type": "image"}),
    ]

    chunks = list(generate_answer_stream(llm, "What shape is shown?", context_docs))

    assert chunks == ["It's ", "a ", "red ", "square."]
    assert "".join(chunks) == "It's a red square."

    sent_content = llm.last_messages[0].content
    assert any(part.get("type") == "text" and "shapes" in part.get("text", "") for part in sent_content)
    image_parts = [part for part in sent_content if part.get("type") == "image_url"]
    assert len(image_parts) == 1
