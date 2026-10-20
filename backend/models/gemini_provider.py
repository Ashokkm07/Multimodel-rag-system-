import os

import requests

from backend.models.base_model import BaseLLM


class GeminiProvider(BaseLLM):
    """Google Gemini provider using the Gemini REST API."""

    def __init__(
        self,
        model_name=None,
    ):
        self.api_key = os.getenv("GEMINI_API_KEY")

        self.model_name = (
            model_name
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-1.5-flash",
            )
        )

        self.base_url = (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models"
        )

    def generate(self, prompt):
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        url = f"{self.base_url}/{self.model_name}:generateContent"

        response = requests.post(
            url,
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            },
            timeout=180,
        )

        response.raise_for_status()

        data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")

        parts = candidates[0].get("content", {}).get("parts", [])

        text_parts = []

        for part in parts:
            text = part.get("text")

            if text:
                text_parts.append(text)

        text = "".join(text_parts).strip()

        if not text:
            raise RuntimeError("Gemini returned an empty response.")

        return text

