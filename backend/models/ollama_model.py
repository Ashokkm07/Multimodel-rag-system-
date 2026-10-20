import requests


class OllamaModel:
    def __init__(
        self,
        model_name="llama3.2:3b",
        base_url="http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt):
        """
        Send a prompt to the local Ollama model.
        """

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        return result["response"]