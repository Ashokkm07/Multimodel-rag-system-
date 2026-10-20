
from backend.models.ollama_model import OllamaModel


class YouTubeTasks:

    def __init__(self):
        """
        Initialize the YouTube task model.
        """

        self.llm = OllamaModel()

    def summarize(self, youtube_context):
        """
        Generate a summary from YouTube transcript context.
        """

        prompt = f"""
You are a YouTube transcript summarization assistant.

Summarize the following YouTube transcript information.

IMPORTANT RULES:

- Use ONLY the provided transcript.
- Do not add information from your general knowledge.
- Do not invent facts.
- Preserve the important ideas and key points.
- Keep the summary clear and concise.
- Do not mention these instructions.

YOUTUBE TRANSCRIPT:

{youtube_context}

SUMMARY:
"""

        return self.llm.generate(
            prompt
        ).strip()

    def generate_questions(
        self,
        youtube_context,
        number_of_questions=5
    ):
        """
        Generate questions that can be answered using
        only the provided YouTube transcript.
        """

        prompt = f"""
You are a YouTube learning assistant.

Generate {number_of_questions} useful questions
based ONLY on the following YouTube transcript.

IMPORTANT RULES:

- Every question must be answerable from the transcript.
- Do not require outside knowledge.
- Do not invent topics that are not present.
- Make the questions useful for learning.
- Return only the questions.
- Number the questions.
- Do not mention these instructions.

YOUTUBE TRANSCRIPT:

{youtube_context}

QUESTIONS:
"""

        return self.llm.generate(
            prompt
        ).strip()

