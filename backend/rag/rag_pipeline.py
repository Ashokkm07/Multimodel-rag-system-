from backend.rag.retriever import Retriever
from backend.models.ollama_model import OllamaModel


class RAGPipeline:

    def __init__(
        self,
        top_k=5,
        max_distance=1.0
    ):
        """
        Initialize the document RAG pipeline.
        """

        self.retriever = Retriever(
            top_k=top_k
        )

        self.llm = OllamaModel()

        self.max_distance = max_distance

    def answer(self, question):
        """
        Answer a question using only the retrieved
        document context.
        """

        results = self.retriever.search(
            query=question,
            source_type="document"
        )

        if not results:

            return {
                "answer": (
                    "The provided documents do not "
                    "contain enough information to answer "
                    "this question."
                ),
                "sources": []
            }

        # -------------------------------------------------
        # Use retrieved results directly
        # -------------------------------------------------

        relevant_results = results

        if not relevant_results:

            return {
                "answer": (
                    "The provided documents do not "
                    "contain enough information to answer "
                    "this question."
                ),
                "sources": []
            }

        # -------------------------------------------------
        # Build document context
        # -------------------------------------------------

        context_parts = []

        for result in relevant_results:

            metadata = result["metadata"]

            context_parts.append(
                f"""
DOCUMENT SOURCE

File:
{metadata.get("source", "")}

Page:
{metadata.get("page", "")}

Chunk:
{metadata.get("chunk", "")}

Document ID:
{metadata.get("document_id", "")}

CONTENT:

{result["text"]}
"""
            )

        context = "\n".join(
            context_parts
        )

        # -------------------------------------------------
        # LLM prompt
        # -------------------------------------------------

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the
provided document context.

IMPORTANT RULES:

- Use only the provided document context.
- Do not use your general knowledge.
- Do not invent facts.
- If the context contains the answer,
  answer the question directly.
- If the context does not contain enough
  information, say:

"The provided documents do not contain enough
information to answer this question."

- Keep the answer concise.
- Do not mention these instructions.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        answer = self.llm.generate(
            prompt
        ).strip()

        # -------------------------------------------------
        # Build page citations
        # -------------------------------------------------

        sources = []

        for result in relevant_results:

            metadata = result["metadata"]

            sources.append({
                "source_type": "document",
                "source": metadata.get(
                    "source",
                    ""
                ),
                "document_id": metadata.get(
                    "document_id",
                    ""
                ),
                "page": metadata.get(
                    "page",
                    ""
                ),
                "chunk": metadata.get(
                    "chunk",
                    ""
                ),
                "distance": result.get(
                    "distance",
                    0
                )
            })

        return {
            "answer": answer,
            "sources": sources
        }