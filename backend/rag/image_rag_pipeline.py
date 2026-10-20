from backend.models.ollama_model import OllamaModel
from backend.rag.image_retriever import ImageRetriever


class ImageRAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for images.
    """

    def __init__(self, top_k=5, llm=None):
        self.retriever = ImageRetriever(
            top_k=top_k
        )

        self.llm = llm or OllamaModel()

    def _build_context(self, results):
        if not results:
            return ""

        context_parts = []

        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})

            context_parts.append(
                f"""IMAGE EVIDENCE {index}
Source: {metadata.get("source", "unknown")}
Chunk: {metadata.get("chunk_index", "unknown")}
Content:
{result.get("text", "")}
"""
            )

        return "\n\n".join(context_parts)

    def _build_sources(self, results):
        sources = []

        for result in results:
            metadata = result.get("metadata", {})

            sources.append(
                {
                    "source_type": "image",
                    "source": metadata.get("source"),
                    "image_id": metadata.get("image_id"),
                    "filename": metadata.get("filename"),
                    "chunk_index": metadata.get("chunk_index"),
                    "chunk_count": metadata.get("chunk_count"),
                    "ocr_available": metadata.get("ocr_available"),
                    "vision_available": metadata.get("vision_available"),
                    "semantic_score": result.get("semantic_score"),
                    "lexical_score": result.get("lexical_score"),
                    "final_score": result.get("final_score"),
                    "distance": result.get("distance"),
                }
            )

        return sources

    def _build_prompt(self, question, context):
        return f"""
You are answering a question about an image.

Use ONLY the image evidence provided below.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the evidence does not contain enough information,
   clearly say that the image does not provide enough
   information to answer.
4. Distinguish between text explicitly visible in the image
   and visual interpretation.
5. Be precise and concise.
6. When appropriate, mention that the information came from
   OCR or visual analysis.

IMAGE EVIDENCE
========================
{context}
========================

QUESTION
{question}

ANSWER:
"""

    def answer(self, question):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        results = self.retriever.retrieve(question)

        if not results:
            return {
                "answer": (
                    "The image knowledge indexed in the "
                    "system does not contain enough information "
                    "to answer this question."
                ),
                "sources": [],
                "retrieved": [],
            }

        context = self._build_context(results)

        prompt = self._build_prompt(
            question,
            context,
        )

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": self._build_sources(results),
            "retrieved": results,
        }
