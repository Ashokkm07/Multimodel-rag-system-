import os

import requests

from backend.models.base_model import BaseLLM


class OpenRouterProvider(BaseLLM):
    """
    OpenRouter provider using its OpenAI-compatible API.
    """

    def __init__(
        self,
        model_name=None,
        base_url="https://openrouter.ai/api/v1/chat/completions",
    ):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.model_name = (
            model_name
            or os.getenv("OPENROUTER_MODEL", "openai/gpt-5")
        )

        self.base_url = base_url

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured.")

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            },
            timeout=180,
        )

        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError("OpenRouter returned no choices.")

        content = choices[0].get("message", {}).get("content", "")

        if not content:
            raise RuntimeError("OpenRouter returned an empty response.")

        return content.strip()