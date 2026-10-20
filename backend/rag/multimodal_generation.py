from backend.models.llm_service import LLMService
from backend.rag.multimodal_router import MultimodalRouter


class MultimodalRAGGenerator:
    """
    Unified generation layer for Document, Image, and YouTube.

    The actual LLM provider is selected through LLMService,
    so this class is independent of Ollama, OpenRouter,
    Gemini, or GPT.
    """

    def __init__(
        self,
        top_k=5,
        llm=None,
        provider=None,
        model_name=None,
    ):
        self.router = MultimodalRouter(top_k=top_k)

        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMService(
                provider=provider,
                model_name=model_name,
            )

    def _build_context(self, results):
        if not results:
            return ""

        context_parts = []

        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})

            source_type = metadata.get("source_type", "unknown")
            source = metadata.get(
                "source",
                metadata.get("filename", "unknown"),
            )
            chunk_index = metadata.get("chunk_index", "unknown")
            timestamp = metadata.get("timestamp")
            timestamp_info = f"\nTimestamp: {timestamp}" if timestamp else ""

            page = metadata.get("page")
            page_info = f"\nPage: {page}" if page is not None else ""

            context_parts.append(
                f"""EVIDENCE {index}
Source type: {source_type}
Source: {source}
Chunk: {chunk_index}{timestamp_info}{page_info}

Content:
{result.get("text", "")}
"""
            )

        return "\n\n".join(context_parts)

    def _build_prompt(self, question, context):
        return f"""
You are the generation component of a grounded multimodal RAG system.

Answer the user's question using ONLY the retrieved evidence below.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the retrieved evidence is insufficient, clearly say that there is not enough information.
4. Keep the answer directly related to the question.
5. Combine multiple evidence items when they support the same answer.
6. Do not claim information is present unless it appears in the evidence.
7. Do not fabricate filenames, page numbers, timestamps, code lines, or citations.

# RETRIEVED EVIDENCE

# {context}

QUESTION
{question}

ANSWER:
"""

    def _build_sources(self, results):
        sources = []

        for result in results:
            metadata = result.get("metadata", {})

            sources.append(
                {
                    "source_type": metadata.get("source_type"),
                    "source": metadata.get(
                        "source",
                        metadata.get("filename"),
                    ),
                    "filename": metadata.get("filename"),
                    "chunk_index": metadata.get("chunk_index"),
                    "page": metadata.get("page"),
                    "start_time": metadata.get("start_time"),
                    "end_time": metadata.get("end_time"),
                    "timestamp": metadata.get("timestamp"),
                    "document_id": metadata.get("document_id"),
                    "image_id": metadata.get("image_id"),
                    "video_id": metadata.get("video_id"),
                    "semantic_score": result.get("semantic_score"),
                    "lexical_score": result.get("lexical_score"),
                    "final_score": result.get("final_score"),
                    "distance": result.get("distance"),
                }
            )

        return sources

    def answer(
        self,
        question,
        source_type="all",
    ):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        retrieval = self.router.retrieve(
            question,
            source_type=source_type,
        )

        results = retrieval.get("results", [])

        if not results:
            return {
                "answer": (
                    "The indexed knowledge does not contain enough information "
                    "to answer this question."
                ),
                "question": question,
                "source_type": source_type,
                "sources": [],
                "retrieved": [],
            }

        context = self._build_context(results)
        prompt = self._build_prompt(question, context)
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "question": question,
            "source_type": source_type,
            "sources": self._build_sources(results),
            "retrieved": results,
        }