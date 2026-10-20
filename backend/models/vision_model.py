
import base64
from io import BytesIO

import requests


class VisionModel:

    def __init__(
        self,
        model_name="llava:7b",
        base_url="http://localhost:11434"
    ):
        """
        Initialize the Ollama vision model.

        Default model:
            llava:7b
        """

        self.model_name = model_name
        self.base_url = base_url

    # =====================================================
    # IMAGE → BASE64
    # =====================================================

    def _image_to_base64(
        self,
        image
    ):
        """
        Convert a PIL image into a base64-encoded PNG.
        """

        buffer = BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode(
            "utf-8"
        )

    # =====================================================
    # IMAGE ANALYSIS
    # =====================================================

    def analyze(
        self,
        image,
        prompt
    ):
        """
        Send an image + prompt to Ollama vision model.
        """

        if image is None:
            raise ValueError(
                "Image cannot be None."
            )

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        image_base64 = (
            self._image_to_base64(
                image
            )
        )

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "images": [
                    image_base64
                ],
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        vision_response = result.get(
            "response",
            ""
        ).strip()

        if not vision_response:
            raise RuntimeError(
                "Ollama returned an empty vision response."
            )

        return vision_response

