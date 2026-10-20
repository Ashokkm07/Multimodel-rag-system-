import re

from backend.embeddings.embedder import Embedder
from backend.vectorstore.chroma_store import ChromaStore


class YouTubeRetriever:

    def __init__(
        self,
        top_k=5,
        candidate_k=15
    ):
        self.embedder = Embedder()

        self.store = ChromaStore()

        self.top_k = top_k

        self.candidate_k = max(
            candidate_k,
            top_k
        )

    # =====================================================
    # TOKENIZATION
    # =====================================================

    def _tokenize(self, text):

        if not text:
            return set()

        text = text.lower()

        return set(
            re.findall(
                r"[a-zA-Z0-9\u0B80-\u0BFF_]+",
                text
            )
        )

    # =====================================================
    # LEXICAL SCORE
    # =====================================================

    def _lexical_score(
        self,
        question,
        text
    ):

        question_tokens = self._tokenize(
            question
        )

        text_tokens = self._tokenize(
            text
        )

        if not question_tokens:
            return 0.0

        overlap = (
            question_tokens
            &
            text_tokens
        )

        return (
            len(overlap)
            /
            len(question_tokens)
        )

    # =====================================================
    # NORMALIZE CHROMA DISTANCE
    # =====================================================

    def _normalize_distances(
        self,
        results
    ):

        if not results:
            return results

        distances = [
            result["distance"]
            for result in results
            if result.get("distance") is not None
        ]

        if not distances:
            return results

        minimum = min(
            distances
        )

        maximum = max(
            distances
        )

        for result in results:

            distance = result.get(
                "distance"
            )

            if distance is None:

                semantic_score = 0.0

            elif minimum == maximum:

                semantic_score = 1.0

            else:

                semantic_score = (
                    1.0
                    -
                    (
                        distance
                        - minimum
                    )
                    /
                    (
                        maximum
                        - minimum
                    )
                )

            result[
                "semantic_score"
            ] = semantic_score

        return results

    # =====================================================
    # SEARCH YOUTUBE ONLY
    # =====================================================

    def search(
        self,
        question
    ):

        if not question or not question.strip():

            return []

        question = question.strip()

        # -------------------------------------------------
        # Generate query embedding
        # -------------------------------------------------

        query_embedding = (
            self.embedder.embed_text(
                question
            )
        )

        # -------------------------------------------------
        # Retrieve more candidates than we finally return
        # -------------------------------------------------

        results = (
            self.store.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=self.candidate_k,
                where={
                    "source_type": "youtube"
                }
            )
        )

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

        candidates = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            # ---------------------------------------------
            # Safety check
            # ---------------------------------------------

            if metadata.get(
                "source_type"
            ) != "youtube":

                continue

            lexical_score = (
                self._lexical_score(
                    question,
                    document
                )
            )

            candidates.append({
                "text": document,
                "metadata": metadata,
                "distance": distance,
                "lexical_score": lexical_score
            })

        # -------------------------------------------------
        # Normalize semantic relevance
        # -------------------------------------------------

        candidates = (
            self._normalize_distances(
                candidates
            )
        )

        # -------------------------------------------------
        # Final reranking
        # -------------------------------------------------

        for candidate in candidates:

            semantic_score = candidate.get(
                "semantic_score",
                0.0
            )

            lexical_score = candidate.get(
                "lexical_score",
                0.0
            )

            candidate[
                "final_score"
            ] = (
                0.70 * semantic_score
                +
                0.30 * lexical_score
            )

        candidates.sort(
            key=lambda item: item[
                "final_score"
            ],
            reverse=True
        )

        return candidates[
            :self.top_k
        ]