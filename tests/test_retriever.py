from langchain_core.documents import Document

from rag.retriever import retrieve


class FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorstore:
    def __init__(self):
        self.calls = []

    def similarity_search_by_vector(self, embedding, k, filter=None):
        self.calls.append((embedding, k, filter))
        element_type = filter["element_type"]
        content = "first" if element_type == "text" else "/tmp/img.png"
        return [Document(page_content=content, metadata={"element_type": element_type})]


def test_retrieve_embeds_query_and_delegates_to_vectorstore():
    embedder = FakeEmbedder()
    vectorstore = FakeVectorstore()

    results = retrieve(vectorstore, embedder, "what shape is shown?", k=2)

    assert vectorstore.calls == [
        ([0.1, 0.2, 0.3], 1, {"element_type": "text"}),
        ([0.1, 0.2, 0.3], 1, {"element_type": "image"}),
    ]
    assert len(results) == 2
    assert results[0].metadata["element_type"] == "text"
    assert results[1].metadata["element_type"] == "image"
