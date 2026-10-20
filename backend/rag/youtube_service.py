
from backend.ingestion.youtube_ingestion_pipeline import (
    YouTubeIngestionPipeline
)

from backend.rag.youtube_rag_pipeline import (
    YouTubeRAGPipeline
)

from backend.rag.youtube_tasks import (
    YouTubeTasks
)

from backend.rag.youtube_result_formatter import (
    YouTubeResultFormatter
)


class YouTubeService:

    def __init__(self):
        """
        Initialize all YouTube RAG components.
        """

        self.ingestion = YouTubeIngestionPipeline()

        self.rag = YouTubeRAGPipeline()

        self.tasks = YouTubeTasks()

    # -------------------------------------------------
    # INGEST YOUTUBE VIDEO
    # -------------------------------------------------

    def ingest(self, youtube_url):
        """
        Process and store a YouTube video.

        This includes:
        - URL validation
        - Video ID extraction
        - Duplicate detection
        - Transcript extraction
        - Text cleaning
        - Chunking
        - Embedding generation
        - ChromaDB storage
        - SQLite status tracking
        """

        return self.ingestion.ingest(
            youtube_url
        )

    # -------------------------------------------------
    # QUESTION ANSWERING
    # -------------------------------------------------

    def ask(self, question):
        """
        Ask a question about the stored YouTube content.

        Returns:
        - Answer
        - YouTube source
        - Video ID
        - Timestamp
        - Distance
        """

        if not question or not question.strip():

            return {
                "answer": "Please provide a question.",
                "sources": []
            }

        result = self.rag.answer(
            question.strip()
        )

        return YouTubeResultFormatter.format_result(
            answer=result["answer"],
            sources=result["sources"]
        )

    # -------------------------------------------------
    # RETRIEVE YOUTUBE CONTEXT
    # -------------------------------------------------

        def get_context(
        self,
        question
    ):

            if not question or not question.strip():
             return ""

        results = self.rag.retriever.search(
            question.strip()
        )

        if not results:
            return ""

        context_parts = []

        for result in results:

            metadata = result[
                "metadata"
            ]

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
                f"{YouTubeResultFormatter.format_timestamp(start_time)}"
                f" - "
                f"{YouTubeResultFormatter.format_timestamp(end_time)}"
            )

            context_parts.append(
                f"""
SOURCE:
{metadata.get('source', '')}

VIDEO ID:
{metadata.get('video_id', '')}

TIMESTAMP:
{timestamp}

TRANSCRIPT:
{result['text']}
"""
            )

        return "\n".join(
            context_parts
        )
    # -------------------------------------------------
    # QUESTION GENERATION
    # -------------------------------------------------

    def generate_questions(
        self,
        question,
        number_of_questions=5
    ):
        """
        Generate learning questions from relevant
        YouTube transcript context.
        """

        context = self.get_context(
            question
        )

        if not context:

            return (
                "The provided YouTube videos do not "
                "contain enough information to generate "
                "questions."
            )

        return self.tasks.generate_questions(
            youtube_context=context,
            number_of_questions=number_of_questions
        )

