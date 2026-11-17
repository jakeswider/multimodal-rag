from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_postgres import PGVector

from ingest.embedder import embed_image_path, embed_text_query
from ingest.pdf_parser import DocElement


def get_vectorstore(
    embedder: OpenCLIPEmbeddings,
    connection_string: str,
    collection_name: str = "multimodal_rag",
) -> PGVector:
    return PGVector(
        embeddings=embedder,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )


def add_elements(
    vectorstore: PGVector,
    embedder: OpenCLIPEmbeddings,
    elements: list[DocElement],
) -> None:
    if not elements:
        return

    texts = [el.content for el in elements]
    metadatas = [
        {
            "source_pdf": el.source_pdf,
            "page_number": el.page_number,
            "element_type": el.element_type,
        }
        for el in elements
    ]
    embeddings = [
        embed_text_query(embedder, el.content)
        if el.element_type == "text"
        else embed_image_path(embedder, el.content)
        for el in elements
    ]

    vectorstore.add_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas)
