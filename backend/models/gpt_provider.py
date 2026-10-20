import os

import requests

from backend.models.base_model import BaseLLM


class GPTProvider(BaseLLM):
    """
    OpenAI GPT provider using the Responses API.
    """

    def __init__(
        self,
        model_name=None,
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.model_name = (
            model_name
            or os.getenv("OPENAI_MODEL", "gpt-5")
        )

        self.base_url = "https://api.openai.com/v1/responses"

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": prompt,
            },
            timeout=180,
        )

        response.raise_for_status()

        data = response.json()

        output_text = data.get("output_text", "").strip()

        if not output_text:
            raise RuntimeError("OpenAI returned an empty response.")

        return output_text