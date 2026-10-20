from backend.models.ollama_model import OllamaModel


class ImageTasks:

    def __init__(self):
        self.llm = OllamaModel()

    def summarize(self, image_context):
        """
        Generate a concise summary from image-derived content.
        """

        prompt = f"""
Summarize the following information extracted from an image.

Use ONLY the provided information.
Do not add facts that are not present.

IMAGE CONTENT:
{image_context}

SUMMARY:
"""

        return self.llm.generate(prompt).strip()

    def generate_questions(
        self,
        image_context,
        number_of_questions=5
    ):
        """
        Generate questions that can be answered from
        the image-derived content.
        """

        prompt = f"""
Generate {number_of_questions} useful questions
that can be answered using ONLY the following
image-derived information.

Do not create questions requiring information
outside the provided content.

IMAGE CONTENT:
{image_context}

QUESTIONS:
"""

        response = self.llm.generate(prompt).strip()

        return response