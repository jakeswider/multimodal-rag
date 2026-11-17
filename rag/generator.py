import base64
import io

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from PIL import Image

MAX_IMAGE_DIMENSION = 512


def build_llm(model: str = "qwen2.5vl") -> ChatOllama:
    return ChatOllama(model=model)


def _encode_image(image_path: str) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_messages(question: str, context_docs: list[Document]) -> list[HumanMessage]:
    content: list[dict] = [
        {
            "type": "text",
            "text": f"Answer the question using only the provided context.\n\nQuestion: {question}\n\nContext:",
        }
    ]

    for doc in context_docs:
        if doc.metadata.get("element_type") == "text":
            content.append({"type": "text", "text": doc.page_content})
        elif doc.metadata.get("element_type") == "image":
            image_b64 = _encode_image(doc.page_content)
            content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"})

    return [HumanMessage(content=content)]


def generate_answer(llm: ChatOllama, question: str, context_docs: list[Document]) -> str:
    response = llm.invoke(_build_messages(question, context_docs))
    return response.content


def generate_answer_stream(llm: ChatOllama, question: str, context_docs: list[Document]):
    for chunk in llm.stream(_build_messages(question, context_docs)):
        if chunk.content:
            yield chunk.content
