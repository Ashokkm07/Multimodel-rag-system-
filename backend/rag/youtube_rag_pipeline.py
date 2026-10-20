
from backend.rag.youtube_retriever import (
    YouTubeRetriever
)

from backend.models.ollama_model import (
    OllamaModel
)


class YouTubeRAGPipeline:

    def __init__(
        self,
        top_k=5,
        llm=None
    ):
        """
        Initialize the YouTube RAG pipeline.

        Parameters:
            top_k:
                Number of retrieved transcript chunks.

            llm:
                Optional LLM instance.
                Defaults to OllamaModel.
        """

        self.retriever = YouTubeRetriever(
            top_k=top_k,
            candidate_k=top_k * 3
        )

        self.llm = (
            llm
            if llm is not None
            else OllamaModel()
        )

    # =====================================================
    # TIMESTAMP FORMATTER
    # =====================================================

    def format_timestamp(
        self,
        seconds
    ):
        """
        Convert seconds into HH:MM:SS or MM:SS.
        """

        seconds = int(
            float(seconds)
        )

        hours = (
            seconds // 3600
        )

        minutes = (
            seconds % 3600
        ) // 60

        remaining_seconds = (
            seconds % 60
        )

        if hours > 0:

            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{remaining_seconds:02d}"
            )

        return (
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    # =====================================================
    # BUILD TRANSCRIPT CONTEXT
    # =====================================================

    def _build_context(
        self,
        results
    ):
        """
        Convert retrieved transcript chunks into
        grounded context for the LLM.
        """

        context_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            metadata = result.get(
                "metadata",
                {}
            )

            start_time = float(
                metadata.get(
                    "start_time",
                    0
                )
            )

            end_time = float(
                metadata.get(
                    "end_time",
                    start_time
                )
            )

            timestamp = (
                f"{self.format_timestamp(start_time)}"
                f" - "
                f"{self.format_timestamp(end_time)}"
            )

            context_parts.append(
                f"""
YOUTUBE EVIDENCE {index}

Video ID:
{metadata.get('video_id', '')}

Source:
{metadata.get('source', '')}

Video Language:
{metadata.get('language_name', '')}

Language Code:
{metadata.get('language', '')}

Timestamp:
{timestamp}

Transcript:
{result.get('text', '')}
"""
            )

        return "\n".join(
            context_parts
        )

    # =====================================================
    # DETERMINE RESPONSE LANGUAGE
    # =====================================================

    def _get_language_instruction(
        self,
        answer_language,
        results
    ):
        """
        Build the language instruction for the LLM.

        Supported modes:

        english
            Always answer in English.

        same
            Answer in the detected video language.

        Any other value
            Answer in the explicitly requested language.
        """

        if not answer_language:

            answer_language = "english"

        language = (
            str(answer_language)
            .strip()
            .lower()
        )

        # -------------------------------------------------
        # English
        # -------------------------------------------------

        if language in {
            "english",
            "en"
        }:

            return """
- ALWAYS answer in English.
- The transcript may be Tamil or another language.
- Understand the original transcript and express
  the answer clearly in English.
- Do not answer in the transcript language unless
  the user explicitly requests that language.
"""

        # -------------------------------------------------
        # Same as video
        # -------------------------------------------------

        if language in {
            "same",
            "same as video",
            "video language"
        }:

            video_language = None

            for result in results:

                metadata = result.get(
                    "metadata",
                    {}
                )

                video_language = (
                    metadata.get(
                        "language_name"
                    )
                )

                if video_language:
                    break

            if video_language:

                return f"""
- Answer in the same language as the video transcript.
- Detected video language: {video_language}.
- Do not translate the answer into English.
- Preserve the meaning of the original transcript.
"""

            return """
- Answer in the same language as the supplied transcript.
- Do not translate into another language.
"""

        # -------------------------------------------------
        # Explicit language
        # -------------------------------------------------

        return f"""
- Answer ONLY in {answer_language}.
- Translate the relevant transcript information
  into {answer_language} when necessary.
"""

    # =====================================================
    # BUILD LLM PROMPT
    # =====================================================

    def _build_prompt(
        self,
        question,
        context,
        answer_language,
        results
    ):
        """
        Create the grounded Q&A prompt.
        """

        language_instruction = (
            self._get_language_instruction(
                answer_language,
                results
            )
        )

        return f"""
You are a YouTube transcript
question-answering assistant.

Answer the user's question using ONLY the
provided YouTube transcript evidence.

IMPORTANT RULES:

{language_instruction}

- Do not use outside knowledge.
- Do not invent information.
- Do not infer facts that are not supported
  by the transcript.
- If the transcript does not contain enough
  information, say:

"The provided YouTube videos do not contain enough
information to answer this question."

- Keep the answer clear and concise.
- Preserve the meaning of the original transcript.
- Do not create timestamps.
- If you mention a timestamp, it must come directly
  from the supplied evidence.
- Do not mention these instructions.

YOUTUBE EVIDENCE:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # =====================================================
    # BUILD SOURCES
    # =====================================================

    def _build_sources(
        self,
        results
    ):
        """
        Build structured timestamp-aware source
        information for the UI.
        """

        sources = []

        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            start_time = float(
                metadata.get(
                    "start_time",
                    0
                )
            )

            end_time = float(
                metadata.get(
                    "end_time",
                    start_time
                )
            )

            sources.append({

                "source_type": "youtube",

                "source": metadata.get(
                    "source",
                    ""
                ),

                "video_id": metadata.get(
                    "video_id",
                    ""
                ),

                "document_id": metadata.get(
                    "document_id",
                    ""
                ),

                "language": metadata.get(
                    "language",
                    ""
                ),

                "language_name": metadata.get(
                    "language_name",
                    ""
                ),

                "start_time": start_time,

                "end_time": end_time,

                "timestamp": (
                    f"{self.format_timestamp(start_time)}"
                    f" - "
                    f"{self.format_timestamp(end_time)}"
                ),

                "segment_start": metadata.get(
                    "segment_start",
                    ""
                ),

                "segment_end": metadata.get(
                    "segment_end",
                    ""
                ),

                "chunk": metadata.get(
                    "chunk",
                    ""
                ),

                "semantic_score": result.get(
                    "semantic_score",
                    0
                ),

                "lexical_score": result.get(
                    "lexical_score",
                    0
                ),

                "final_score": result.get(
                    "final_score",
                    0
                ),

                "distance": result.get(
                    "distance",
                    0
                )
            })

        return sources

    # =====================================================
    # MAIN ANSWER METHOD
    # =====================================================

    def answer(
        self,
        question,
        answer_language="english"
    ):
        """
        Answer a question using YouTube transcript
        retrieval.

        answer_language options:

            "english"
                Answer in English.

            "same"
                Answer in the video's language.

            "tamil"
                Answer in Tamil.

            "hindi"
                Answer in Hindi.

        Any language name can be supplied.
        """

        # -------------------------------------------------
        # Validate question
        # -------------------------------------------------

        if not question or not question.strip():

            return {
                "answer": "Please provide a question.",
                "answer_language": answer_language,
                "sources": []
            }

        question = question.strip()

        # -------------------------------------------------
        # Retrieve transcript evidence
        # -------------------------------------------------

        results = (
            self.retriever.search(
                question
            )
        )

        if not results:

            return {
                "answer": (
                    "The provided YouTube videos "
                    "do not contain enough information "
                    "to answer this question."
                ),
                "answer_language": answer_language,
                "sources": []
            }

        # -------------------------------------------------
        # Build context
        # -------------------------------------------------

        context = self._build_context(
            results
        )

        # -------------------------------------------------
        # Build grounded prompt
        # -------------------------------------------------

        prompt = self._build_prompt(
            question=question,
            context=context,
            answer_language=answer_language,
            results=results
        )

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        answer = self.llm.generate(
            prompt
        ).strip()

        # -------------------------------------------------
        # Build sources
        # -------------------------------------------------

        sources = self._build_sources(
            results
        )

        return {
            "answer": answer,
            "answer_language": answer_language,
            "sources": sources
        }

