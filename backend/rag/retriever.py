from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore


class Retriever:

    def __init__(self, top_k=5):
        """
        Initialize the embedding model and ChromaDB.
        """

        self.embedder = Embedder()
        self.store = ChromaStore()
        self.top_k = top_k

    def search(self, query, source_type=None):
        """
        Search ChromaDB for chunks relevant to the query.

        source_type can be:
        - None
        - document
        - image
        - youtube
        - github
        """

        query_embedding = self.embedder.embed_text(
            query
        )

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": self.top_k
        }

        # Apply source filtering when requested
        if source_type:
            query_kwargs["where"] = {
                "source_type": source_type
            }

        results = self.store.collection.query(
            **query_kwargs
        )

        retrieved_chunks = []

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            retrieved_chunks.append({
                "text": document,
                "metadata": metadata,
                "distance": distance
            })

        return retrieved_chunks