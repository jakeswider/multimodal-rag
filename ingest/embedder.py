from langchain_experimental.open_clip import OpenCLIPEmbeddings


def get_embedder() -> OpenCLIPEmbeddings:
    return OpenCLIPEmbeddings(model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k")


def embed_text_query(embedder: OpenCLIPEmbeddings, text: str) -> list[float]:
    return embedder.embed_query(text)


def embed_image_path(embedder: OpenCLIPEmbeddings, image_path: str) -> list[float]:
    return embedder.embed_image([image_path])[0]