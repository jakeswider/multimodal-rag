from langchain_core.documents import Document


def retrieve(vectorstore, embedder, query: str, k: int = 4) -> list[Document]:
    query_embedding = embedder.embed_query(query)
    k_text = max(1, k // 2)
    k_image = k - k_text

    text_docs = vectorstore.similarity_search_by_vector(
        query_embedding, k=k_text, filter={"element_type": "text"}
    )
    image_docs = vectorstore.similarity_search_by_vector(
        query_embedding, k=k_image, filter={"element_type": "image"}
    )

    return text_docs + image_docs
