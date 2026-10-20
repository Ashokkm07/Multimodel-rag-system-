import re

from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore

class ImageRetriever:
    """
    Retrieves image knowledge from ChromaDB.


    Retrieval is restricted to source_type=image so that
    image questions do not retrieve documents, YouTube
    transcripts, or GitHub code.
    """

    def __init__(self, top_k=5, candidate_k=15):
        self.top_k = top_k
        self.candidate_k = candidate_k

        self.embedder = Embedder()
        self.store = ChromaStore()

    def _tokenize(self, text):
        if not text:
            return set()

        return set(
            re.findall(
                r"[A-Za-z0-9_]+|[\u0B80-\u0BFF]+",
                text.lower(),
            )
        )

    def _lexical_score(self, query, text):
        query_tokens = self._tokenize(query)
        text_tokens = self._tokenize(text)

        if not query_tokens or not text_tokens:
            return 0.0

        overlap = query_tokens.intersection(text_tokens)

        return len(overlap) / len(query_tokens)

    def _semantic_score(self, distance):
        if distance is None:
            return 0.0

        distance = max(float(distance), 0.0)

        return 1.0 / (1.0 + distance)

    def _embed_query(self, query):
        """
        Uses the existing Embedder interface.

        The project Embedder exposes embed_documents(),
        so a single query is embedded as a one-item document batch.
        """

        embeddings = self.embedder.embed_documents([query])

        if not embeddings:
            raise RuntimeError(
                "Embedder returned no embedding for the query."
            )

        return embeddings[0]

    def retrieve(self, query):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self._embed_query(query)

        results = self.store.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.candidate_k,
            where={"source_type": "image"},
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]

        metadatas = results.get("metadatas", [[]])[0]

        distances = results.get("distances", [[]])[0]

        candidates = []

        for index, document in enumerate(documents):
            metadata = (
                metadatas[index] if index < len(metadatas) else {}
            )

            distance = (
                distances[index] if index < len(distances) else None
            )

            semantic_score = self._semantic_score(distance)

            lexical_score = self._lexical_score(query, document)

            final_score = 0.70 * semantic_score + 0.30 * lexical_score

            candidates.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                    "semantic_score": semantic_score,
                    "lexical_score": lexical_score,
                    "final_score": final_score,
                }
            )

        candidates.sort(
            key=lambda item: item["final_score"],
            reverse=True,
        )

        return candidates[:self.top_k]

